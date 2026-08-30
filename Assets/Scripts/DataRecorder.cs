using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using System.Xml.Serialization;

public class DataRecorder : MonoBehaviour
{
    [System.Serializable]
    public class BoneTransformData
    {
        public string boneName;
        public Vector3 localPosition;
        public Quaternion localRotation;
    }

    [System.Serializable]
    public class HandFrameData
    {
        public float timeStamp;
        public List<BoneTransformData> bonesData = new List<BoneTransformData>();
    }

    [System.Serializable]
    public class LetterDataset
    {
        [XmlAttribute("Letter")]
        public string letter; // Força a serialização como texto "A" em vez de valor numérico/char

        public List<HandFrameData> capturedFrames = new List<HandFrameData>();
    }

    [XmlRoot("LibrasDataset")]
    [System.Serializable]
    public class DatasetContainer
    {
        [XmlArray("Datasets")]
        [XmlArrayItem("LetterData")]
        public List<LetterDataset> datasets = new List<LetterDataset>();
    }

    [SerializeField] private TextMeshProUGUI statusText;
    [SerializeField] private List<Transform> rootBones = new List<Transform>();
    [SerializeField] private float captureDurationSeconds = 10f;
    [SerializeField] private int targetCapturesCount = 100;
    [SerializeField] private char currentLetter = 'A';
    [SerializeField] private string xmlFileName = "LibrasDataset.xml";

    [SerializeField] private List<LetterDataset> recordedData = new List<LetterDataset>();

    private List<Transform> allBones = new List<Transform>();
    private bool isRecording = false;

    private void Start()
    {
        InitializeBones();
        UpdateStatusText();
    }

    public void InitializeBones()
    {
        allBones.Clear();
        foreach (Transform root in rootBones)
        {
            if (root != null)
                CollectBonesRecursive(root);
        }
    }

    private void CollectBonesRecursive(Transform parent)
    {
        if (parent.name.ToLower().Contains("velocity")) return;

        allBones.Add(parent);

        foreach (Transform child in parent)
        {
            CollectBonesRecursive(child);
        }
    }

    public void StartRecording()
    {
        if (!isRecording)
        {
            InitializeBones();
            StartCoroutine(CaptureRoutine());
        }
    }

    private IEnumerator CaptureRoutine()
    {
        isRecording = true;

        int validCaptures = Mathf.Max(1, targetCapturesCount);
        float intervalInSeconds = captureDurationSeconds / validCaptures;

        LetterDataset dataset = new LetterDataset { letter = currentLetter.ToString() };

        for (int i = 0; i < validCaptures; i++)
        {
            float currentTime = i * intervalInSeconds;

            if (statusText != null)
                statusText.text = $"Capturing letter {currentLetter}... ({i + 1}/{validCaptures})";

            HandFrameData frame = new HandFrameData { timeStamp = currentTime };

            foreach (Transform bone in allBones){
                if (bone == null || bone.name.ToLower().Contains("velocity")) continue;

                frame.bonesData.Add(new BoneTransformData
                {
                    boneName = bone.name,
                    localPosition = bone.localPosition,
                    localRotation = bone.localRotation
                });
            }

            dataset.capturedFrames.Add(frame);

            yield return new WaitForSeconds(intervalInSeconds);
        }

        int existingIndex = recordedData.FindIndex(d => d.letter == currentLetter.ToString());
        if (existingIndex >= 0)
        {
            recordedData[existingIndex] = dataset;
        }
        else
        {
            recordedData.Add(dataset);
        }

        isRecording = false;
        NextLetter();
    }

    public void ChangeLetter(int step)
    {
        if (isRecording) return;

        int newChar = currentLetter + step;
        if (newChar < 'A') newChar = 'Z';
        if (newChar > 'Z') newChar = 'A';
        currentLetter = (char)newChar;

        UpdateStatusText();
    }

    public void NextLetter() => ChangeLetter(1);
    public void PreviousLetter() => ChangeLetter(-1);

    public void UpdateStatusText()
    {
        if (statusText != null)
        {
            statusText.text = $"Ready to capture letter {currentLetter}";
        }
    }

    public void ExportToXML()
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;

        string targetFolder = Path.Combine(projectRoot, "Data");
        if (!Directory.Exists(targetFolder))
        {
            Directory.CreateDirectory(targetFolder);
        }

        string filePath = Path.Combine(targetFolder, xmlFileName);

        DatasetContainer container = new DatasetContainer
        {
            datasets = recordedData
        };

        XmlSerializer serializer = new XmlSerializer(typeof(DatasetContainer));
        using (FileStream stream = new FileStream(filePath, FileMode.Create))
        {
            serializer.Serialize(stream, container);
        }

        Debug.Log($"Dados exportados com sucesso para: {filePath}");
    }
}