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


    [Header("Regras de Infer�ncia")]
    [Tooltip("Porcentagem m�nima de confian�a (0 a 100) para aceitar a predi��o e exibir a letra.")]
    [Range(0f, 100f)]
    [SerializeField] private float minConfidenceThreshold = 70f;

    [Header("Debug / Diagnostico")]
    [Tooltip("Caminho do CSV gerado pelo DataRecorder (relativo a pasta do projeto, a que contem 'Assets'), usado so para comparar as colunas de treino com as da cena atual. Se nao encontrar o arquivo, so a lista da cena e impressa.")]
    [SerializeField] private string trainingCsvRelativePath = "Assets/Data/LibrasDataset.csv";
    [SerializeField] private bool useMockData = false;

    [Header("Normalizacao (StandardScaler do Colab)")]
    private readonly float[] featureMeans = new float[] { 0.0f, 0.0f, 0.0f, 1.0f, -0.017149171428571426f, -0.012695077040816327f, 0.038183939285714286f, -0.023758000000000005f, -0.007381f, 0.09684000000000002f, 0.08035957653061226f, 0.040349586224489795f, 0.023969847959183672f, 0.9830599280612244f, 0.3248948326530612f, 0.039893064795918365f, -0.02327452244897959f, 0.8381327479591837f, 0.05625836275510205f, 0.04426085357142857f, -0.009018792857142856f, 0.7487592147959183f, 0.17085120306122448f, 0.06268914285714286f, -0.1422582755102041f, 0.954545355612245f, 0.4787085984693878f, 0.032816918367346945f, -0.21847393265306123f, 0.7559868428571428f, 0.32085137346938775f, -0.019882131632653063f, -0.16535749540816325f, 0.5702055158163265f, -0.000797234693877551f, -0.009267867857142858f, 0.03600247244897959f, -0.0017409999999999997f, -0.002565423469387755f, 0.09648788724489794f, 0.24034457755102043f, 0.057014640306122445f, -0.021052892346938776f, 0.9570833f, 0.5656458642857143f, 0.03884072602040817f, -0.055855946428571425f, 0.7000817801020408f, 0.16254676989795913f, 0.034272726530612245f, -0.039382488775510205f, 0.5721129887755101f, -0.0012691469387755103f, -0.005916684693877551f, 0.06624501581632654f, 0.2290227943877551f, 0.05998361275510204f, -0.06836102397959183f, 0.9546237163265306f, 0.5494707933673469f, 0.017383895408163264f, -0.12964768265306123f, 0.7002979086734694f, 0.14215984693877545f, 0.008524710714285713f, -0.06306472448979591f, 0.5489441765306122f, -0.02479523112244898f, -0.022437817346938775f, 0.037631786734693876f, 0.0668086413265306f, -0.3386364571428572f, 0.6669700020408162f, 0.6284891913265307f, 0.24932844642857144f, -0.09903130102040815f, 0.6715428974489797f, 0.6493236357142858f };
    private readonly float[] featureScales = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 0.0010667888871730518f, 0.0008784694555707871f, 0.000376806038568294f, 1.0f, 1.0f, 1.0f, 0.14239463071386513f, 0.06490795760591407f, 0.0073750322945917565f, 0.019734640655674485f, 0.3569965119081334f, 0.06169088223388681f, 0.03225259544086251f, 0.23989890493888733f, 0.5479431242478563f, 0.0673045573467452f, 0.051117795661109744f, 0.3560477643145668f, 0.17247942991502146f, 0.05330785902985543f, 0.027657531973864165f, 0.04615361975300163f, 0.31665813272623333f, 0.03413195353259903f, 0.0460910162623751f, 0.2166858047867427f, 0.6019945938380452f, 0.10115828331096495f, 0.16019013624846895f, 0.3819564087905573f, 0.0010667865555596702f, 0.00043921949836311567f, 0.00037679801639513516f, 1.0f, 1.0f, 1.0f, 0.14285458683678762f, 0.02380411095662714f, 0.011452063163248948f, 0.03777814470539056f, 0.33608790506067215f, 0.019790105139992967f, 0.028952658433484265f, 0.2666778985791089f, 0.707978796391997f, 0.04799342363555604f, 0.10555512877001266f, 0.3589718790950206f, 0.0005333923622273435f, 0.00021961851100935802f, 0.00018839740957220526f, 0.15683836589028127f, 0.032765760017136054f, 0.014892689276886641f, 0.04557909395080899f, 0.34419955631957383f, 0.022358847133142717f, 0.033853464185839885f, 0.26539583281375667f, 0.7119776475618194f, 0.046315754678205576f, 0.13705173753598776f, 0.3828327335249804f, 0.002133569205401005f, 0.0017569285964911811f, 0.0007536046332749152f, 0.13775087251571305f, 0.11358668912555875f, 0.03856392282697778f, 0.0874654785847118f, 0.10857080734540875f, 0.17094663994977366f, 0.06407239836292011f, 0.10159619989639732f };

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
        "A", "B"
        , "C", "D",// "E", "F", "G", "I", "L", "M",
       // "N", "O", "P", "Q", "R", "S", "T", "U", "V"
    };

    private void Start()
    {
        // O modelo precisa ser carregado ANTES de montar o mapeamento de features,
        // pois � dele que tiramos a quantidade esperada (nao existe mais numero fixo).
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
    /// da ultima dimensao, que � o numero de features que o modelo espera.
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
            Debug.LogError("[HandInference] A ultima dimensao do input do modelo � dinamica (sem valor fixo). " +
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
            Debug.LogWarning($"[HandInference] DIVERG�NCIA: Cena montou {featureMapping.Count} features. O script ajustar� automaticamente para as {expectedFeatureCount} primeiras!");
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

        if (!Directory.Exists(dataFolderPath))
        {
            Debug.LogWarning($"[HandInference] Pasta 'Data' n?o encontrada em: {dataFolderPath}");
            return;
        }

        string[] csvFiles = Directory.GetFiles(dataFolderPath, "*.csv");

        if (csvFiles.Length == 0)
        {
            Debug.LogWarning($"[HandInference] Nenhum arquivo .csv encontrado em: {dataFolderPath}");
            return;
        }

        string csvFullPath = csvFiles[0];

        string headerLine;
        using (StreamReader reader = new StreamReader(csvFullPath))
        {
            headerLine = reader.ReadLine();
        }

        if (string.IsNullOrEmpty(headerLine))
        {
            Debug.LogWarning($"[HandInference] CSV de treino ({Path.GetFileName(csvFullPath)}) est? vazio ou sem cabe?alho.");
            return;
        }

        List<string> csvColumns = headerLine.Split(',').ToList();
        if (csvColumns.Count > 0 && csvColumns[csvColumns.Count - 1] == "Letter")
            csvColumns.RemoveAt(csvColumns.Count - 1);

        // 3) Comparacao por CONJUNTO (nomes presentes, ignora ordem)
        List<string> inSceneNotInCsv = sceneFeatures.Where(f => !csvColumns.Contains(f)).ToList();
        List<string> inCsvNotInScene = csvColumns.Where(f => !sceneFeatures.Contains(f)).ToList();

        bool isMatch = (inSceneNotInCsv.Count == 0 && inCsvNotInScene.Count == 0 && sceneFeatures.Count == csvColumns.Count);

        string matchStatus = isMatch ? "<color=green>SIM (Match perfeito!)</color>" : "<color=red>N?O (H? diverg?ncias)</color>";

        Debug.Log($"[HandInference] Quantidade CENA: {sceneFeatures.Count} | Quantidade CSV: {csvColumns.Count} | Deu Match (conjunto)? {matchStatus}");

        if (inSceneNotInCsv.Count > 0)
            Debug.LogWarning($"[HandInference] Presentes na CENA mas AUSENTES no CSV ({inSceneNotInCsv.Count}):\n" + string.Join("\n", inSceneNotInCsv));

        if (inCsvNotInScene.Count > 0)
            Debug.LogWarning($"[HandInference] Presentes no CSV mas AUSENTES na CENA ({inCsvNotInScene.Count}):\n" + string.Join("\n", inCsvNotInScene));

        // 4) Comparacao por ORDEM (posicao a posicao) -- e' o que realmente importa pro vetor de entrada
        bool orderMatches = sceneFeatures.Count == csvColumns.Count;
        if (!orderMatches)
        {
            Debug.LogError($"[DIAGNOSTICO ORDEM] Tamanhos diferentes: cena tem {sceneFeatures.Count}, CSV tem {csvColumns.Count}. Nao da pra comparar posicao a posicao.");
        }
        else
        {
            int mismatches = 0;
            for (int i = 0; i < sceneFeatures.Count; i++)
            {
                if (sceneFeatures[i] != csvColumns[i])
                {
                    mismatches++;
                    Debug.LogError($"[DIAGNOSTICO ORDEM] Divergencia na posicao {i}: cena tem '{sceneFeatures[i]}', CSV tem '{csvColumns[i]}'");
                    orderMatches = false;
                }
            }
            if (mismatches > 0)
                Debug.LogError($"[DIAGNOSTICO ORDEM] Total de posicoes divergentes: {mismatches} de {sceneFeatures.Count}.");
        }

        Debug.Log(orderMatches
            ? "<color=green>[DIAGNOSTICO ORDEM] Ordem BATE perfeitamente (posicao a posicao).</color>"
            : "<color=red>[DIAGNOSTICO ORDEM] Ordem NAO bate -- veja os erros acima. Isso corrompe o vetor de entrada mesmo com 'match' de conjunto.</color>");
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

        Transform wristTransform = (rootBones != null && rootBones.Count > 0) ? rootBones[0] : null;

        if (wristTransform == null)
        {
            Debug.LogError("[HandInference] WristTransform nao atribuido!");
            return;
        }
        //Debug.Log($"[CHECK] featureMapping.Count={featureMapping.Count} expectedFeatureCount={expectedFeatureCount}");

        for (int i = 0; i < featureMapping.Count; i++)
        {
            var (bone, comp) = featureMapping[i];

            if (bone == null) continue;

            if (bone == wristTransform)
            {
                if (comp < 3)
                {
                    rawCollectedFeatures[i] = 0f;
                }
                else
                {
                    Quaternion relativeRot = Quaternion.Inverse(wristTransform.rotation) * bone.rotation;
                    relativeRot = LibrasBoneConfig.ExtractSwing(relativeRot, Vector3.forward);

                    if (relativeRot.w < 0)
                    {
                        relativeRot.x = -relativeRot.x;
                        relativeRot.y = -relativeRot.y;
                        relativeRot.z = -relativeRot.z;
                        relativeRot.w = -relativeRot.w;
                    }

                    rawCollectedFeatures[i] = GetQuaternionComponent(relativeRot, comp - 3);
                }
            }
            else
            {
                Vector3 relativePos = wristTransform.InverseTransformPoint(bone.position);
                Quaternion relativeRot = Quaternion.Inverse(wristTransform.rotation) * bone.rotation;

                if (relativeRot.w < 0)
                {
                    relativeRot.x = -relativeRot.x;
                    relativeRot.y = -relativeRot.y;
                    relativeRot.z = -relativeRot.z;
                    relativeRot.w = -relativeRot.w;
                }

                if (comp < 3)
                {
                    rawCollectedFeatures[i] = comp switch
                    {
                        0 => relativePos.x,
                        1 => relativePos.y,
                        2 => relativePos.z,
                        _ => 0f
                    };
                }
                else
                {
                    int rotComp = comp - 3;
                    rawCollectedFeatures[i] = GetQuaternionComponent(relativeRot, rotComp);
                }
            }

            //Debug.Log($"[CHECK] featureMapping.Count={featureMapping.Count} expectedFeatureCount={expectedFeatureCount}");
            //Debug.Log("[FULL] " + string.Join(",", rawCollectedFeatures));
        }

        int copyLength = Math.Min(featureMapping.Count, expectedFeatureCount);

        for (int i = 0; i < copyLength; i++)
        {
            float rawVal = rawCollectedFeatures[i];

            if (featureMeans != null && featureMeans.Length > i &&
                featureScales != null && featureScales.Length > i)
            {
                float scale = featureScales[i] != 0f ? featureScales[i] : 1f;
                inputFeatures[i] = (rawVal - featureMeans[i]) / scale;
            }
            else
            {
                inputFeatures[i] = rawVal;
            }
        }
    }

    // M�todo auxiliar para ler componentes do Quaternion de forma segura
    private float GetQuaternionComponent(Quaternion q, int rotComp)
    {
        return rotComp switch
        {
            0 => q.x,
            1 => q.y,
            2 => q.z,
            3 => q.w,
            _ => 0f
        };
    }

    private string Predict()
    {
        if (worker == null || expectedFeatureCount <= 0)
            return GetDebugRandomLetter();

        TensorShape shape = new TensorShape(1, expectedFeatureCount);
        using Tensor<float> inputTensor = new Tensor<float>(shape, inputFeatures);

        worker.Schedule(inputTensor);

        using Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

        if (outputTensor == null) return "";

        var readOnlyArray = outputTensor.DownloadToArray();
        for (int i = 0; i < Math.Min(readOnlyArray.Length, logitsBuffer.Length); i++)
        {
            logitsBuffer[i] = readOnlyArray[i];
        }

        // Estabilidade num�rica do Softmax
        float maxLogit = float.MinValue;
        for (int i = 0; i < labels.Length; i++)
        {
            if (logitsBuffer[i] > maxLogit)
                maxLogit = logitsBuffer[i];
        }

        float sumExp = 0f;
        for (int i = 0; i < labels.Length; i++)
        {
            probsBuffer[i] = Mathf.Exp(logitsBuffer[i] - maxLogit);
            sumExp += probsBuffer[i];
        }

        int predictedClassIndex = 0;
        float maxProb = -1f;

        for (int i = 0; i < labels.Length; i++)
        {
            if (sumExp > 0f)
                probsBuffer[i] = (probsBuffer[i] / sumExp) * 100f;
            else
                probsBuffer[i] = 0f;

            if (probsBuffer[i] > maxProb)
            {
                maxProb = probsBuffer[i];
                predictedClassIndex = i;
            }
        }

        string predictedClass = (predictedClassIndex < labels.Length) ? labels[predictedClassIndex] : "";
        float confidence = maxProb;

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

        float sumExp = 0f;
        for (int i = 0; i < labels.Length; i++)
            sumExp += probsBuffer[i];

        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        for (int i = 0; i < labels.Length; i++)
        {
            float probability = sumExp > 0f ? (probsBuffer[i] / sumExp) * 100f : 0f;
            sb.AppendLine($"{labels[i]} - {probability:F1}%");
        }
        allLabelsListText.text = sb.ToString();
    }
}