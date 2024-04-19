import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def GiB(val):
    return val * 1 << 30


def build_engine(onnx_file_path, precision='fp32', dynamic_shapes=None):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""

    EXPLICIT_BATCH_FLAG = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH_FLAG)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse model file
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loading ONNX file from path {onnx_file_path}...')
    with open(onnx_file_path, 'rb') as model:
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')

    encoder_input_index = None
    decoder_input_index = None
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if input_tensor.name == "encoder_input":
            encoder_input_index = i
        elif input_tensor.name == "decoder_input":
            decoder_input_index = i

    if encoder_input_index is None or decoder_input_index is None:
        raise ValueError("Could not find encoder or decoder input in the network")
    else:
        print(f"encoder_index -> {encoder_input_index}, decoder_index -> {decoder_input_index}")

    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Input number: {network.num_inputs}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Output number: {network.num_outputs}')

    if dynamic_shapes is not None:
        profile = builder.create_optimization_profile()

        encoder_input = network.get_input(encoder_input_index)
        encoder_shape = (1, 128)
        profile.set_shape(encoder_input.name, encoder_shape, encoder_shape, encoder_shape)

        decoder_input = network.get_input(decoder_input_index)
        decoder_min_shape = (1, 1)
        decoder_opt_shape = dynamic_shapes['opt_shape']
        decoder_max_shape = (1, 128)
        profile.set_shape(decoder_input.name, decoder_min_shape, decoder_opt_shape, decoder_max_shape)

        config.add_optimization_profile(profile)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(1))  # 1G

    if precision == 'fp32':
        pass
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
    else:
        raise ValueError('precision must be one of fp32, fp16, or int8')

    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building an engine from file {onnx_file_path}; this may take a while...')
    serialized_engine = builder.build_serialized_network(network, config)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed creating Engine')

    with open(onnx_file_path, "wb") as f:
        f.write(serialized_engine)

    return serialized_engine


def save_engine(_engine, path):
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Saving engine to file {path}')
    with open(path, 'wb') as f:
        f.write(_engine)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed saving engine')


if __name__ == "__main__":
    model_path = r"D:\pycharm_projects\MachineLearning\transformer\export\No-En-Transformer.onnx"
    shape_dict = {"opt_shape": (1, 64)}
    engine = build_engine(model_path, precision="fp16", dynamic_shapes=shape_dict)
    print(engine)
    save_engine(engine, r"D:\pycharm_projects\MachineLearning\transformer\tensorRT\engine.trt")
