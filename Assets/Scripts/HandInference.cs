using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Unity.InferenceEngine;
using System;

public class HandInference : MonoBehaviour
{
    [Header("Configuracoes do Modelo Sentis")]
    [SerializeField] private ModelAsset modelAsset;

    [Header("UI & Referencias")]
    [SerializeField] private TextMeshProUGUI resultText;
    [SerializeField] private TextMeshProUGUI confidenceText;
    [SerializeField] private TextMeshProUGUI buttonLabelText;
    [SerializeField] private List<Transform> rootBones = new List<Transform>();
    [SerializeField] private float inferenceInterval = 0.5f;

    private List<Transform> allBones = new List<Transform>();
    private float[] inputFeatures;
    private string lastLetter = "";
    private bool isTranslating = false;
    private Coroutine inferenceCoroutine;

    private Model runtimeModel;
    private Worker worker;

    // Buffers reutilizaveis para evitar o Garbage Collector / Memory Leak
    private float[] logitsBuffer;
    private float[] probsBuffer;
    private HashSet<Transform> visitedBones = new HashSet<Transform>();

    private readonly string[] labels = new string[] {
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

        foreach (Transform root in rootBones)
        {
            if (root != null)
                CollectBonesRecursive(root);
        }

        int totalFeatures = 0;
        foreach (Transform bone in allBones)
        {
            if (bone == null) continue;

            bool isWrist = bone.name.ToLower().Contains("wrist");
            totalFeatures += isWrist ? 4 : 7;
        }

        inputFeatures = new float[totalFeatures];
    }

    private void CollectBonesRecursive(Transform parent)
    {
        if (parent == null || visitedBones.Contains(parent)) return;
        visitedBones.Add(parent);

        if (!ShouldIgnoreBone(parent.name))
            allBones.Add(parent);

        foreach (Transform child in parent)
            CollectBonesRecursive(child);
    }

    private bool ShouldIgnoreBone(string boneName)
    {
        if (string.IsNullOrEmpty(boneName)) return true;
        string n = boneName.ToLower();
        return n.Contains("velocity") || n.Contains("tip");
    }

    private IEnumerator InferenceRoutine()
    {
        while (isTranslating){
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
        int index = 0;
        for (int i = 0; i < allBones.Count; i++)
        {
            Transform bone = allBones[i];
            if (bone == null) continue;

            string boneName = bone.name;
            Vector3 pos = bone.localPosition;
            Quaternion rot = bone.localRotation;

            bool isWrist = boneName.ToLower().Contains("wrist");

            if (!isWrist)
            {
                inputFeatures[index++] = pos.x;
                inputFeatures[index++] = pos.y;
                inputFeatures[index++] = pos.z;
            }

            inputFeatures[index++] = rot.x;
            inputFeatures[index++] = rot.y;
            inputFeatures[index++] = rot.z;
            inputFeatures[index++] = rot.w;
        }
    }

    private string Predict()
    {
        if (worker == null)
            return GetDebugRandomLetter();

        // 1. Cria o Tensor temporario descartando os recursos da GPU via 'using'
        TensorShape shape = new TensorShape(1, 1, inputFeatures.Length);
        using Tensor<float> inputTensor = new Tensor<float>(shape, inputFeatures);

        worker.Schedule(inputTensor);

        // 2. Le a saida do worker sem criar leak de tensor
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        if (outputTensor == null) return "";

        // Copia os dados para o buffer sem instanciar novos objetos no C#
        var readOnlyArray = outputTensor.DownloadToArray();
        for (int i = 0; i < Math.Min(readOnlyArray.Length, logitsBuffer.Length); i++)
        {
            logitsBuffer[i] = readOnlyArray[i];
        }

        // 3. ArgMax (classe com maior score)
        int predictedClassIndex = 0;
        float maxLogit = float.MinValue;

        for (int i = 0; i < logitsBuffer.Length; i++)
        {
            if (logitsBuffer[i] > maxLogit)
            {
                maxLogit = logitsBuffer[i];
                predictedClassIndex = i;
            }
        }

        string predictedClass = "";
        if (predictedClassIndex < labels.Length)
            predictedClass = labels[predictedClassIndex];

        // 4. Softmax
        float sumExp = 0f;
        for (int i = 0; i < logitsBuffer.Length; i++)
        {
            probsBuffer[i] = Mathf.Exp(logitsBuffer[i] - maxLogit);
            sumExp += probsBuffer[i];
        }

        float confidence = (probsBuffer[predictedClassIndex] / sumExp) * 100f;

        if (confidenceText != null)
            confidenceText.text = $"Classe: {predictedClass} - {confidence:F1}%";

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