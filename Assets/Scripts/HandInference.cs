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

    private void OnDestroy(){
        worker?.Dispose();
    }

    private void InitializeModel()
    {
        if (modelAsset != null){
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = new Worker(runtimeModel, BackendType.GPUCompute);
        }
        else{
            Debug.LogError("InferenceEngine: Nenhum arquivo .onnx atribuído!");
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
        foreach (Transform root in rootBones)
        {
            if (root != null)
                CollectBonesRecursive(root);
        }

        // Calcula a quantidade de atributos dinamicamente
        int totalFeatures = 0;
        foreach (Transform bone in allBones)
        {
            if (bone == null) continue;

            bool isWrist = bone.name.ToLower().Contains("wrist");

            // Se não for o Wrist, conta 3 (Posição) + 4 (Rotação) = 7
            // Se for o Wrist, conta apenas 4 (Rotação)
            totalFeatures += isWrist ? 4 : 7;
        }

        inputFeatures = new float[totalFeatures];
    }

    private void CollectBonesRecursive(Transform parent)
    {
        if (!ShouldIgnoreBone(parent.name))
        {
            allBones.Add(parent);
        }
        foreach (Transform child in parent)
        {
            CollectBonesRecursive(child);
        }
    }

    private bool ShouldIgnoreBone(string boneName)
    {
        if (string.IsNullOrEmpty(boneName)) return true;
        string n = boneName.ToLower();
        return n.Contains("velocity") || n.Contains("tip");
    }

    private IEnumerator InferenceRoutine()
    {
        while (isTranslating)
        {
            CollectFeatures();

            //ALTERNAR ENTRE O PLACEHOLDER E A INFERENCIA
            string predictedLetter = Predict(inputFeatures);
            //string predictedLetter = GetDebugRandomLetter();

            if (!string.IsNullOrEmpty(predictedLetter) && predictedLetter != lastLetter)
            {
                lastLetter = predictedLetter;

                if (resultText != null)
                    resultText.text += lastLetter;
            }

            yield return new WaitForSeconds(inferenceInterval);
        }
    }

    private void CollectFeatures(){
        int index = 0;
        for (int i = 0; i < allBones.Count; i++){
            Transform bone = allBones[i];
            if (bone == null) continue;

            string boneName = bone.name;

            Vector3 pos = bone.localPosition;
            Quaternion rot = bone.localRotation;

            // Se for o primeiro osso (Wrist/Pulso), ignora as posicoes XYZ igual feito no Python:
            // df.drop(columns=['R_Wrist_PosX', 'R_Wrist_PosY', 'R_Wrist_PosZ'])
            bool isWrist = boneName.ToLower().Contains("wrist");

            if (!isWrist)
            {
                inputFeatures[index++] = pos.x;
                inputFeatures[index++] = pos.y;
                inputFeatures[index++] = pos.z;
            }

            // Adiciona as rotacoes para todos os bones
            inputFeatures[index++] = rot.x;
            inputFeatures[index++] = rot.y;
            inputFeatures[index++] = rot.z;
            inputFeatures[index++] = rot.w;
        }
    }

    private string Predict(float[] features){
        if (worker == null)
            return GetDebugRandomLetter();

        // 1. Executa a inferência na GPU via Sentis
        TensorShape shape = new TensorShape(1, 1, features.Length);
        using Tensor<float> inputTensor = new Tensor<float>(shape, features);

        worker.Schedule(inputTensor);

        using Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        float[] logits = outputTensor.DownloadToArray();

        // 2. Encontra a classe com maior pontuação bruta (ArgMax)
        int predictedClassIndex = 0;
        float maxLogit = float.MinValue;

        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit)
            {
                maxLogit = logits[i];
                predictedClassIndex = i;
            }
        }

        string predictedClass = "";
        if (predictedClassIndex < labels.Length)
            predictedClass = labels[predictedClassIndex];

        // 3. Aplica o Softmax para converter os Logits em Probabilidade (%)
        float sumExp = 0f;
        float[] probs = new float[logits.Length];

        for (int i = 0; i < logits.Length; i++){
            probs[i] = Mathf.Exp(logits[i] - maxLogit); // Evita overflow numérico
            sumExp += probs[i];
        }

        float confidence = (probs[predictedClassIndex] / sumExp) * 100f;

        // 4. Atualiza o log na UI com a porcentagem
        if (confidenceText != null)
            confidenceText.text = $"Classe: {predictedClass} - {confidence:F1}%";

        return predictedClass;
    }

    private string GetDebugRandomLetter()
    {
        char randomChar = (char)UnityEngine.Random.Range('A', 'Z' + 1);
        return randomChar.ToString();
    }

    public void ClearText()
    {
        lastLetter = "";
        if (resultText != null)
            resultText.text = "";
        if (confidenceText != null)
            confidenceText.text = "";
    }
}