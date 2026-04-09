import onnx


def print_onnx_info(path):
    model = onnx.load(path)
    print(f"\n📄 模型: {path}")
    print("🔹 输入节点:")
    for inp in model.graph.input:
        dims = [
            d.dim_value if d.dim_value != 0 else d.dim_param
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"  {inp.name}: {dims}")
    print("🔸 输出节点:")
    for out in model.graph.output:
        dims = [
            d.dim_value if d.dim_value != 0 else d.dim_param
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"  {out.name}: {dims}")


# 替换为你的文件路径
print_onnx_info("det_model.onnx")
print_onnx_info("rec_model.onnx")
