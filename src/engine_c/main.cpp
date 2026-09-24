#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    // Creamos el entorno dentro de Ort (Namespace de Windows) y definimos que se use solo 1 hilo del procesador
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "QuantEngine");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1); 
    
    // En Windows, ONNX exige strings de caracteres anchos (wchar_t) para las rutas
    const wchar_t* model_path = L"../models/quant_model.onnx"; 
    
    std::cout << "Cargando grafo..." << std::endl;
    Ort::Session session(env, model_path, session_options);
    
    //Estructuras de memoria consecutiva para los datos
    std::vector<float> input_data = {500.5f, 100.0f, 0.001f, 499.0f, 498.5f}; 
    std::vector<int64_t> input_shape = {1, 5}; 
    
    // Punteros a memoria
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_data.data(), 
        input_data.size(), 
        input_shape.data(), 
        input_shape.size()
    );

    std::cout << "Tensor ONNX creado" << std::endl;
    const char* input_names[] = {"float_input"};
    const char* output_names[] = {"label"}; //'label' es la prediccion final

    // Inferencia
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor, 
        1, 
        output_names, 
        1
    );

    // Lectura de la salida (Puntero)
    // ONNX devuelve un int64_t para las clasificaciones (0 o 1)
    int64_t* output_data = output_tensors.front().GetTensorMutableData<int64_t>();
    
    std::cout << "---------------------------------" << std::endl;
    std::cout << "PREDICCION DEL TICK: " << output_data[0] << std::endl;
    std::cout << "---------------------------------" << std::endl;
    
    return 0;
}