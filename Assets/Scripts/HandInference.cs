using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class HandInference : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI resultText;
    [SerializeField] private TextMeshProUGUI buttonLabelText;
    [SerializeField] private List<Transform> rootBones = new List<Transform>();
    [SerializeField] private float inferenceInterval = 0.5f;

    private List<Transform> allBones = new List<Transform>();
    private float[] inputFeatures;
    private string lastLetter = "";
    private bool isTranslating = false;
    private Coroutine inferenceCoroutine;

    private void Start()
    {
        InitializeBones();

        if (resultText != null)
            resultText.text = "";

        if (buttonLabelText != null)
            buttonLabelText.text = "Start translating";
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

        inputFeatures = new float[allBones.Count * 7];
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
            string predictedLetter = Predict(inputFeatures);

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
        int index = 0;
        for (int i = 0; i < allBones.Count; i++)
        {
            Transform bone = allBones[i];
            if (bone == null) continue;

            Vector3 pos = bone.localPosition;
            Quaternion rot = bone.localRotation;

            inputFeatures[index++] = pos.x;
            inputFeatures[index++] = pos.y;
            inputFeatures[index++] = pos.z;
            inputFeatures[index++] = rot.x;
            inputFeatures[index++] = rot.y;
            inputFeatures[index++] = rot.z;
            inputFeatures[index++] = rot.w;
        }
    }

    private string Predict(float[] features)
    {
        // TODO: Integrar com Sentis/Barracuda ou execucao do modelo exportado (.onnx, .tflite, etc)
        // Passar os dados contidos no array 'features' para o modelo e retornar a letra inferida.

        return GetDebugRandomLetter();
    }

    private string GetDebugRandomLetter()
    {
        char randomChar = (char)Random.Range('A', 'Z' + 1);
        return randomChar.ToString();
    }

    public void ClearText()
    {
        lastLetter = "";
        if (resultText != null)
            resultText.text = "";
    }
}