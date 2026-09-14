"""让 tests/ 在没装 torch 的机器上也能跑纯逻辑测试。

折切分、Rank IC、HAC 统计都是纯 numpy 逻辑,不该被 2GB 的 torch 依赖挡住,
这样 CI 里能直接跑。已装 torch 则原样使用,不做任何替换。

哑模块只满足 import 期的需要(类继承、装饰器),不模拟任何数值行为。
"""

import importlib.util
import sys
import types


class _DummyMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _make_dummy(name)


class _Dummy(metaclass=_DummyMeta):
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        # 当作装饰器用时保持原函数:@torch.no_grad()
        return a[0] if len(a) == 1 and callable(a[0]) else None

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _make_dummy(name)


def _make_dummy(name):
    return _DummyMeta(name, (_Dummy,), {})


def _stub(dotted):
    mod = types.ModuleType(dotted)

    def __getattr__(attr):
        if attr.startswith("__") and attr.endswith("__"):
            # __path__/__spec__ 必须缺席,否则 import 机制把哑类型当协议迭代
            raise AttributeError(attr)
        return _make_dummy(attr)

    mod.__getattr__ = __getattr__
    sys.modules[dotted] = mod
    return mod


def install_torch_stub():
    """torch 缺失时注册哑模块;已安装则什么都不做。"""
    if importlib.util.find_spec("torch") is not None:
        return False
    names = ["torch", "torch.nn", "torch.utils", "torch.utils.data",
             "torch.optim", "torch.amp"]
    mods = {n: _stub(n) for n in names}
    mods["torch"].nn = mods["torch.nn"]
    mods["torch"].utils = mods["torch.utils"]
    mods["torch.utils"].data = mods["torch.utils.data"]
    mods["torch.nn"].functional = _make_dummy("functional")
    return True


install_torch_stub()
