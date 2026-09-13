using UnityEngine;

public class FollowCamera : MonoBehaviour
{
    public Camera Camera2Follow;
    public float CameraDistance = 1;
    public float smoothTime = 0.3F;
    private Vector3 velocity = Vector3.zero;
    private Transform target;

    void Awake()
    {
        target = Camera2Follow.transform;
    }

    void Update()
    {
        // Definir distância da camera
        Vector3 targetPosition = target.TransformPoint(new Vector3(0, 0, CameraDistance));

        // Suavizar movimento
        transform.position = Vector3.SmoothDamp(transform.position, targetPosition, ref velocity, smoothTime);

        // versao 1: sempre olhando pra camera
        transform.LookAt(transform.position + Camera2Follow.transform.rotation * Vector3.forward, Camera2Follow.transform.rotation * Vector3.up);

        // versao 2: rotação suave para a camera
        transform.rotation = Quaternion.RotateTowards(transform.rotation, target.rotation, 35 * Time.deltaTime);
    }
}