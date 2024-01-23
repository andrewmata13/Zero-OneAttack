from other_attacks.optimal_attack.auto_LiRPA.bound_ops import *

bound_op_map = {
    # 'onnx::MaxPool': BoundMaxPool2d,
    # 'onnx::GlobalAveragePool': AdaptiveAvgPool2d,
    'onnx::Gemm': BoundLinear,
    'prim::Constant': BoundPrimConstant,
}
