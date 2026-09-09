using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Unity.InferenceEngine;
using System;

public class HandInference : MonoBehaviour
{
    private const int EXPECTED_FEATURE_COUNT = 77;

    [Header("Configuracoes do Modelo Sentis")]
    [SerializeField] private ModelAsset modelAsset;

    [Header("UI & Referencias")]
    [SerializeField] private TextMeshProUGUI resultText;
    [SerializeField] private TextMeshProUGUI confidenceText;
    [SerializeField] private TextMeshProUGUI buttonLabelText;
    [SerializeField] private List<Transform> rootBones = new List<Transform>();
    [SerializeField] private float inferenceInterval = 0.5f;
    [SerializeField] private float delayToStart = 2f;

    [Header("Regras de Inferência")]
    [Tooltip("Porcentagem mínima de confiança (0 a 100) para aceitar a predição e exibir a letra.")]
    [Range(0f, 100f)]
    [SerializeField] private float minConfidenceThreshold = 70f;

    private List<Transform> allBones = new List<Transform>();
    private float[] inputFeatures = new float[EXPECTED_FEATURE_COUNT];
    private float[] rawCollectedFeatures;
    
    private string lastLetter = "";
    private bool isTranslating = false;
    private Coroutine inferenceCoroutine;

    private Model runtimeModel;
    private Worker worker;

    private float[] logitsBuffer;
    private float[] probsBuffer;
    private HashSet<Transform> visitedBones = new HashSet<Transform>();

    private List<(Transform bone, int componentIndex)> featureMapping = new List<(Transform, int)>();

    private static readonly string[] labels = new string[] {
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z"
    };

    private void Start()
    {
        InitializeBones();
        InitializeModel();
        ClearText();

        if (buttonLabelText != null)
            buttonLabelText.text = "Start translating";
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }

    private void InitializeModel()
    {
        if (modelAsset != null)
        {
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = new Worker(runtimeModel, BackendType.GPUCompute);

            logitsBuffer = new float[labels.Length];
            probsBuffer = new float[labels.Length];
        }
        else
        {
            Debug.LogError("InferenceEngine: Nenhum arquivo .onnx atribuido!");
        }
    }

    public void ToggleTranslation()
    {
        isTranslating = !isTranslating;

        if (isTranslating)
        {
            if (buttonLabelText != null)
                buttonLabelText.text = "Stop translating";

            inferenceCoroutine = StartCoroutine(InferenceRoutine());
        }
        else
        {
            if (buttonLabelText != null)
                buttonLabelText.text = "Start translating";

            if (inferenceCoroutine != null)
                StopCoroutine(inferenceCoroutine);
        }
    }

    public void InitializeBones()
    {
        allBones.Clear();
        visitedBones.Clear();
        featureMapping.Clear();

        foreach (Transform root in rootBones)
        {
            if (root != null)
                CollectBonesRecursive(root);
        }

        foreach (Transform bone in allBones)
        {
            if (bone == null) continue;
            string cleanBoneName = bone.name.Replace(",", "_");

            for (int c = 0; c < LibrasBoneConfig.ComponentSuffixes.Length; c++)
            {
                string colName = $"{cleanBoneName}_{LibrasBoneConfig.ComponentSuffixes[c]}";
                if (!LibrasBoneConfig.IgnoredColumns.Contains(colName))
                {
                    featureMapping.Add((bone, c));
                }
            }
        }

        rawCollectedFeatures = new float[featureMapping.Count];

        Debug.Log($"[HandInference] Features encontradas na cena: {featureMapping.Count}. O modelo espera: {EXPECTED_FEATURE_COUNT}.");

        if (featureMapping.Count != EXPECTED_FEATURE_COUNT)
        {
            Debug.LogWarning($"[HandInference] DIVERGÊNCIA: Cenas montaram {featureMapping.Count} features. O script ajustará automaticamente para as {EXPECTED_FEATURE_COUNT} primeiras!");
        }
    }

    private void CollectBonesRecursive(Transform parent)
    {
        if (parent == null || visitedBones.Contains(parent)) return;
        visitedBones.Add(parent);

        if (!LibrasBoneConfig.ShouldIgnoreBone(parent.name))
            allBones.Add(parent);

        foreach (Transform child in parent)
            CollectBonesRecursive(child);
    }

    private IEnumerator InferenceRoutine()
    {
        yield return new WaitForSeconds(delayToStart);
        while (isTranslating)
        {
            CollectFeatures();

            string predictedLetter = Predict();

            if (!string.IsNullOrEmpty(predictedLetter) && predictedLetter != lastLetter){
                lastLetter = predictedLetter;

                if (resultText != null)
                    resultText.text += lastLetter;
            }

            yield return new WaitForSeconds(inferenceInterval);
        }
    }

    private void CollectFeatures()
    {
        // 1. Coleta tudo que encontrou na cena
        for (int i = 0; i < featureMapping.Count; i++){
            var (bone, comp) = featureMapping[i];
            rawCollectedFeatures[i] = LibrasBoneConfig.GetComponentValue(bone, comp);
        }

        // 2. Trava de segurança: Copia estritamente até 77 posições para o tensor
        int copyLength = Math.Min(featureMapping.Count, EXPECTED_FEATURE_COUNT);
        Array.Copy(rawCollectedFeatures, inputFeatures, copyLength);
    }

    private string Predict()
    {
        if (worker == null)
            return GetDebugRandomLetter();

        // Shape sempre garantido como (1, 1, 77)
        TensorShape shape = new TensorShape(1, 1, EXPECTED_FEATURE_COUNT);
        using Tensor<float> inputTensor = new Tensor<float>(shape, inputFeatures);

        worker.Schedule(inputTensor);

        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        if (outputTensor == null) return "";

        var readOnlyArray = outputTensor.DownloadToArray();
        for (int i = 0; i < Math.Min(readOnlyArray.Length, logitsBuffer.Length); i++){
            logitsBuffer[i] = readOnlyArray[i];
        }

        int predictedClassIndex = 0;
        float maxLogit = float.MinValue;

        for (int i = 0; i < logitsBuffer.Length; i++){
            if (logitsBuffer[i] > maxLogit){
                maxLogit = logitsBuffer[i];
                predictedClassIndex = i;
            }
        }

        string predictedClass = "";
        if (predictedClassIndex < labels.Length)
            predictedClass = labels[predictedClassIndex];

        float sumExp = 0f;
        for (int i = 0; i < logitsBuffer.Length; i++){
            probsBuffer[i] = Mathf.Exp(logitsBuffer[i] - maxLogit);
            sumExp += probsBuffer[i];
        }

        float confidence = (probsBuffer[predictedClassIndex] / sumExp) * 100f;

        if (confidenceText != null)
            confidenceText.text = $"Classe: {predictedClass} - {confidence:F1}%";

        if (confidence < minConfidenceThreshold)
            return "";

        return predictedClass;
    }

    private string GetDebugRandomLetter(){
        char randomChar = (char)UnityEngine.Random.Range('A', 'Z' + 1);
        return randomChar.ToString();
    }

    public void ClearText(){
        lastLetter = "";
        if (resultText != null)
            resultText.text = "";
        if (confidenceText != null)
            confidenceText.text = "";
    }
}