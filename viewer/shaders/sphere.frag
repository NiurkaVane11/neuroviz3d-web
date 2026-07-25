#version 330 core
in vec3 FragPos;
in vec3 Normal;
out vec4 FragColor;
uniform vec3 uLightPos;
uniform vec3 uViewPos;
uniform vec3 uColor;
uniform vec3 uEmissive;   // color emisivo aditivo (activacion), default vec3(0.0)
void main() {
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(uLightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 viewDir = normalize(uViewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 ambient = 0.25 * uColor;
    vec3 diffuse = 0.65 * diff * uColor;
    vec3 specular = 0.4 * spec * vec3(1.0);

    vec3 phong = ambient + diffuse + specular;
    vec3 result = phong + uEmissive;

    // Soft clamp: evita quemar a blanco puro, pero permite superar 1.0 un poco (glow suave)
    result = result / (1.0 + max(result - vec3(1.0), vec3(0.0)));

    FragColor = vec4(result, 1.0);
}
