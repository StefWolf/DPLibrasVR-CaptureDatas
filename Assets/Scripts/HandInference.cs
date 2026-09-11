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
    private readonly float[] featureMeans = new float[] { 0.0f, 0.0f, 0.0f, 1.0f, -0.02026140918367347f, -0.009418401020408163f, 0.03717333163265306f, -0.024136024489795918f, -0.007498361224489796f, 0.09838208469387756f, 0.2546110102040816f, 0.056688999999999996f, 0.0069564204081632615f, 0.9189544999999999f, 0.5501665734693877f, 0.036099f, -0.05877899897959184f, 0.5849525000000001f, -0.39442442448979587f, 0.008038802040816322f, 0.05702050000000001f, 0.6604813989795918f, 0.31703749897959177f, 0.09400100510204082f, -0.143925f, 0.8310444867346939f, 0.5584029979591837f, 0.07349749897959183f, -0.20046850000000002f, 0.5626414489795917f, -0.32922150000000006f, 0.15059550000000002f, -0.015115500000000049f, 0.705443418367347f, -0.0036490275510204086f, -0.007676055102040814f, 0.034957164285714284f, -0.0017689999999999997f, -0.002606445918367347f, 0.09802375102040817f, 0.3434215826530612f, 0.087175f, -0.048508525510204084f, 0.8647159999999999f, 0.5736505000000001f, 0.04678400000000001f, -0.08751899999999997f, 0.5346880153061226f, -0.3652160387755103f, 0.029452448979591827f, 0.0449f, 0.7060177408163265f, -0.002708901020408164f, -0.0051412653061224485f, 0.06649047551020407f, 0.33444799999999997f, 0.08236200000000002f, -0.0821612387755102f, 0.8454194999999999f, 0.5546045000000002f, 0.03997508367346937f, -0.13537600000000002f, 0.5386064734693878f, -0.34046550000000003f, 0.06780250000000002f, 0.0248624724489796f, 0.7109419999999997f, -0.03086848571428572f, -0.01583745918367347f, 0.03499382040816326f, -0.10876899897959184f, -0.36238949489795913f, 0.5602669642857142f, 0.7273954928571428f, 0.05736034387755104f, -0.19966150000000005f, 0.5514503765306121f, 0.7931395000000001f };
    private readonly float[] featureScales = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 0.0009501828980315213f, 0.0018299704735543112f, 0.000970691555089373f, 5.372622789211232e-05f, 1.6637740676761274e-05f, 0.00021895339167253387f, 0.2777940102040996f, 0.05784299999999999f, 0.021412440817955358f, 0.08037349999999999f, 0.42435457346946154f, 0.030355f, 0.06061500102041656f, 0.4070835f, 0.5402114244898553f, 0.02530719796154631f, 0.06848650000000002f, 0.328206425510334f, 0.38333449897959315f, 0.09227899489798655f, 0.05308499999999999f, 0.14499523163332953f, 0.4027300020408189f, 0.11352050102041264f, 0.009917499999999997f, 0.38840055102046717f, 0.5236424999999999f, 0.020530500000000007f, 0.1912415f, 0.24466241836748653f, 0.0009870275642079344f, 0.0009215409494176379f, 0.000965720075294352f, 3.999999999999989e-06f, 5.5584219675054624e-06f, 0.00021818683542326693f, 0.3240935826531676f, 0.011537999999999998f, 0.041555525510495364f, 0.13220800000000002f, 0.41184950000000004f, 0.029974000000000008f, 0.057784999999999996f, 0.44869600714287394f, 0.5222369653061569f, 0.05491055102082532f, 0.11717799999999999f, 0.27530973673492415f, 0.0004918786699149763f, 0.00046373278836029746f, 0.0005919628353028968f, 0.36910200000000004f, 0.024347999999999998f, 0.036990238777196434f, 0.14723949999999997f, 0.42591249999999997f, 0.05357408367411961f, 0.03604199999999999f, 0.4436461000002697f, 0.5336815f, 0.0094745f, 0.14201852755111205f, 0.261447f, 0.0019219166015374618f, 0.003653159229836295f, 0.0018537942085751504f, 0.002470001020614302f, 0.10771849489798266f, 0.04255103571467541f, 0.02052150714302871f, 0.035460656123962965f, 0.1460535f, 0.030254376532148897f, 0.013167499999999999f };

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