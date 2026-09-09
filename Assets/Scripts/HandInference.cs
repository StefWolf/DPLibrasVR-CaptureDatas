using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Unity.InferenceEngine;
using System;
using System.IO;
using System.Linq;

public class HandInference : MonoBehaviour
{
    [Header("Configuracoes do Modelo Sentis")]
    [SerializeField] private ModelAsset modelAsset;

    [Header("UI & Referencias")]
    [SerializeField] private TextMeshProUGUI resultText;
    [SerializeField] private TextMeshProUGUI confidenceText;
    [SerializeField] private TextMeshProUGUI buttonLabelText;
    [SerializeField] private TextMeshProUGUI allLabelsListText;
    [SerializeField] private List<Transform> rootBones = new List<Transform>();
    [SerializeField] private float inferenceInterval = 0.5f;
    [SerializeField] private float delayToStart = 2f;


    [Header("Regras de Inferência")]
    [Tooltip("Porcentagem mínima de confiança (0 a 100) para aceitar a predição e exibir a letra.")]
    [Range(0f, 100f)]
    [SerializeField] private float minConfidenceThreshold = 70f;

    [Header("Debug / Diagnostico")]
    [Tooltip("Caminho do CSV gerado pelo DataRecorder (relativo a pasta do projeto, a que contem 'Assets'), usado so para comparar as colunas de treino com as da cena atual. Se nao encontrar o arquivo, so a lista da cena e impressa.")]
    [SerializeField] private string trainingCsvRelativePath = "Assets/Data/LibrasDataset.csv";
    [SerializeField] private bool useMockData = false;

    // Quantidade de features que o modelo espera, lida diretamente do ModelAsset.
    // Deixa de ser um numero fixo/chutado e passa a ser derivada da shape de input do proprio modelo.
    private int expectedFeatureCount = -1;

    private List<Transform> allBones = new List<Transform>();
    private float[] inputFeatures;
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
        "A", "B", "C", "D", "E", "F", "G", "I", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V"
    };

    private void Start()
    {
        // O modelo precisa ser carregado ANTES de montar o mapeamento de features,
        // pois é dele que tiramos a quantidade esperada (nao existe mais numero fixo).
        InitializeModel();
        InitializeBones();
        LogFeatureComparison();
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

            expectedFeatureCount = GetExpectedFeatureCountFromModel(runtimeModel);
            inputFeatures = new float[Mathf.Max(expectedFeatureCount, 0)];

            logitsBuffer = new float[labels.Length];
            probsBuffer = new float[labels.Length];

            Debug.Log($"[HandInference] Modelo carregado.");
        }
        else
        {
            Debug.LogError("InferenceEngine: Nenhum arquivo .onnx atribuido!");
        }
    }

    /// <summary>
    /// Le a shape do primeiro input do modelo (ex.: 1x1x77) e retorna o tamanho
    /// da ultima dimensao, que é o numero de features que o modelo espera.
    /// Substitui o antigo "magic number" fixo (77).
    /// </summary>
    private int GetExpectedFeatureCountFromModel(Model model)
    {
        if (model == null || model.inputs == null || model.inputs.Count == 0)
        {
            Debug.LogError("[HandInference] O modelo nao possui inputs definidos, nao foi possivel determinar a quantidade de features.");
            return 0;
        }

        DynamicTensorShape dynamicShape = model.inputs[0].shape;

        if (dynamicShape.isRankDynamic)
        {
            Debug.LogError("[HandInference] O input do modelo tem rank dinamico, nao foi possivel determinar a quantidade de features.");
            return 0;
        }

        // Get(-1) le apenas a ULTIMA dimensao (a de features), sem exigir que a shape inteira
        // seja estatica. Isso importa porque a dimensao de batch (eixo 0) costuma ser dinamica
        // (ex.: shape "(?, 1, 77)"), o que fazia ToTensorShape() falhar mesmo com as features fixas.
        int lastDimValue = dynamicShape.Get(-1);

        if (lastDimValue < 0)
        {
            Debug.LogError("[HandInference] A ultima dimensao do input do modelo é dinamica (sem valor fixo). " +
                            "Verifique como o modelo foi exportado (o input deveria terminar em um numero fixo de features).");
            return 0;
        }

        return lastDimValue;
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

        //Debug.Log($"[HandInference] Features encontradas na cena: {featureMapping.Count}. O modelo espera: {expectedFeatureCount}.");

        if (featureMapping.Count != expectedFeatureCount)
        {
            Debug.LogWarning($"[HandInference] DIVERGÊNCIA: Cena montou {featureMapping.Count} features. O script ajustará automaticamente para as {expectedFeatureCount} primeiras!");
        }
    }

    /// <summary>
    /// Ferramenta de diagnostico: imprime a lista de features (colunas) que a CENA ATUAL monta
    /// (na mesma ordem usada em CollectFeatures/Predict) e, se encontrar o CSV de treino gerado
    /// pelo DataRecorder, compara as duas listas para apontar exatamente quais colunas
    /// estao sobrando na cena ou faltando em relacao ao que o modelo aprendeu.
    /// </summary>
    [ContextMenu("Log Feature Comparison (Scene vs Training CSV)")]
    public void LogFeatureComparison()
    {
        // 1) Features da CENA
        List<string> sceneFeatures = new List<string>();
        foreach (var (bone, comp) in featureMapping)
        {
            string cleanBoneName = bone != null ? bone.name.Replace(",", "_") : "NULL";
            sceneFeatures.Add($"{cleanBoneName}_{LibrasBoneConfig.ComponentSuffixes[comp]}");
        }

        // 2) Features do CSV DE TREINO
        string dataFolderPath = Path.Combine(Application.dataPath, "Data");

        if (!Directory.Exists(dataFolderPath)){
            Debug.LogWarning($"[HandInference] Pasta 'Data' não encontrada em: {dataFolderPath}");
            return;
        }

        string[] csvFiles = Directory.GetFiles(dataFolderPath, "*.csv");

        if (csvFiles.Length == 0){
            Debug.LogWarning($"[HandInference] Nenhum arquivo .csv encontrado em: {dataFolderPath}");
            return;
        }

        // Pega o primeiro arquivo .csv encontrado na pasta automaticamente
        string csvFullPath = csvFiles[0];

        string headerLine;
        using (StreamReader reader = new StreamReader(csvFullPath)){
            headerLine = reader.ReadLine();
        }

        if (string.IsNullOrEmpty(headerLine)){
            Debug.LogWarning($"[HandInference] CSV de treino ({Path.GetFileName(csvFullPath)}) está vazio ou sem cabeçalho.");
            return;
        }

        List<string> csvColumns = headerLine.Split(',').ToList();
        if (csvColumns.Count > 0 && csvColumns[csvColumns.Count - 1] == "Letter")
            csvColumns.RemoveAt(csvColumns.Count - 1);

        // 3) Comparação e Impressão Resumida
        List<string> inSceneNotInCsv = sceneFeatures.Where(f => !csvColumns.Contains(f)).ToList();
        List<string> inCsvNotInScene = csvColumns.Where(f => !sceneFeatures.Contains(f)).ToList();

        bool isMatch = (inSceneNotInCsv.Count == 0 && inCsvNotInScene.Count == 0 && sceneFeatures.Count == csvColumns.Count);

        string matchStatus = isMatch ? "<color=green>SIM (Match perfeito!)</color>" : "<color=red>NÃO (Há divergências)</color>";

        Debug.Log($"[HandInference] Quantidade CENA: {sceneFeatures.Count} | Quantidade CSV: {csvColumns.Count} | Deu Match? {matchStatus}");

        if (inSceneNotInCsv.Count > 0)
            Debug.LogWarning($"[HandInference] Presentes na CENA mas AUSENTES no CSV ({inSceneNotInCsv.Count}):\n" + string.Join("\n", inSceneNotInCsv));

        if (inCsvNotInScene.Count > 0)
            Debug.LogWarning($"[HandInference] Presentes no CSV mas AUSENTES na CENA ({inCsvNotInScene.Count}):\n" + string.Join("\n", inCsvNotInScene));
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

            if (!string.IsNullOrEmpty(predictedLetter) && predictedLetter != lastLetter)
            {
                lastLetter = predictedLetter;

                if (resultText != null)
                    resultText.text += lastLetter;
            }

            yield return new WaitForSeconds(inferenceInterval);
        }
    }

    private void CollectFeatures()
    {
        if (useMockData) return;

        // 1. Coleta tudo que encontrou na cena
        for (int i = 0; i < featureMapping.Count; i++)
        {
            var (bone, comp) = featureMapping[i];
            rawCollectedFeatures[i] = LibrasBoneConfig.GetComponentValue(bone, comp);
        }

        // 2. Trava de segurança: Copia estritamente até expectedFeatureCount posições para o tensor
        int copyLength = Math.Min(featureMapping.Count, expectedFeatureCount);
        Array.Copy(rawCollectedFeatures, inputFeatures, copyLength);
    }

    private string Predict()
    {
        if (worker == null || expectedFeatureCount <= 0)
            return GetDebugRandomLetter();

        // Shape sempre garantida de acordo com o que o modelo espera (1, 1, expectedFeatureCount)
        TensorShape shape = new TensorShape(1, 1, expectedFeatureCount);
        using Tensor<float> inputTensor = new Tensor<float>(shape, inputFeatures);

        worker.Schedule(inputTensor);

        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        if (outputTensor == null) return "";

        var readOnlyArray = outputTensor.DownloadToArray();
        for (int i = 0; i < Math.Min(readOnlyArray.Length, logitsBuffer.Length); i++)
        {
            logitsBuffer[i] = readOnlyArray[i];
        }

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

        float sumExp = 0f;
        for (int i = 0; i < logitsBuffer.Length; i++)
        {
            probsBuffer[i] = Mathf.Exp(logitsBuffer[i] - maxLogit);
            sumExp += probsBuffer[i];
        }

        float confidence = (probsBuffer[predictedClassIndex] / sumExp) * 100f;

        if (confidenceText != null)
            confidenceText.text = $"Classe: {predictedClass} - {confidence:F1}%";

        UpdateAllLabelsUI();

        if (confidence < minConfidenceThreshold)
            return "";

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

    private void UpdateAllLabelsUI()
    {
        if (allLabelsListText == null) return;

        // Calcula o somatório do denominador do softmax (sumExp) caso ainda não tenha sido feito
        float sumExp = 0f;
        for (int i = 0; i < logitsBuffer.Length; i++)
        {
            sumExp += probsBuffer[i]; // Nota: assume que probsBuffer já guarda os exponenciais calculados no Predict
        }

        System.Text.StringBuilder sb = new System.Text.StringBuilder();

        for (int i = 0; i < labels.Length; i++)
        {
            float probability = 0f;
            if (sumExp > 0f && i < probsBuffer.Length)
            {
                probability = (probsBuffer[i] / sumExp) * 100f;
            }

            sb.AppendLine($"{labels[i]} - {probability:F1}%");
        }

        allLabelsListText.text = sb.ToString();
    }
}