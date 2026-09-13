using UnityEngine;
using UnityEngine.InputSystem; // <-- Necessário para o novo Input System

public class HandInferenceMock : MonoBehaviour
{
    private HandInference handInferenceScript;

    void Start(){
        handInferenceScript = GetComponent<HandInference>();
        if (handInferenceScript == null){
            Debug.LogError("[Mock] O script HandInference não foi encontrado no mesmo GameObject!");
        }
    }

    void Update(){
        InjectFakeData();
    }

    private void InjectFakeData()
    {
        if (handInferenceScript == null) return;

        // Usa Reflection para acessar o array privado 'inputFeatures' e injetar dados aleatórios
        var field = typeof(HandInference).GetField("inputFeatures", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (field != null){
            float[] features = (float[])field.GetValue(handInferenceScript);
            if (features != null){
                for (int i = 0; i < features.Length; i++){
                    features[i] = UnityEngine.Random.Range(-2f, 2f);
                }
                Debug.Log("[Mock] Dados de teste injetados no tensor!");
            }
        }
    }
}