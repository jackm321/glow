from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

# https://our.internmc.facebook.com/intern/anp/view/?id=163884

inputs = torch.LongTensor([0, 31414] + [232, 328] * 24 + [2])

torch_glow.setMinFusionGroupSize(25)
# torch_glow.setFusionBlacklist(["prim::ConstantChunk"])
torch_glow.enableDumpGlowDag()

# def fun(inputs):
#     return m(inputs)

with torch.no_grad():
    torch_glow.disableFusionPass()
    m = torch.jit.load("/Users/jackmontgomery/Desktop/roberta_public.pt1")
    m.eval()
    a = m(inputs)
    # print(m.graph_for(inputs))

    torch_glow.enableFusionPass()

    m2 = torch.jit.load("/Users/jackmontgomery/Desktop/roberta_public.pt1")
    m2.eval()
    print(m2.graph_for(inputs))

    b = m2(inputs)
    print("inputs size:", inputs.size())
    print("jit model: ", a)
    print("glow model: ", b)
    print("dif: ", a-b)

    # print(m(inputs))
    # torch_glow.enableFusionPass()
    # m2 = torch.jit.trace(m, inputs)
    # graph = m2.graph_for(inputs)
    # print(m2(inputs))
    # print(graph)

    # graph = m.graph_for(inputs)
    # print(graph)
    # torch_glow.glowCustomFuseDebug_(graph)
    # print(graph)
    # torch_glow.fuseKnownPatterns(graph)
    # print(graph)
    # m.graph_for(inputs)

# with torch.no_grad():
#     torch_glow.enableFusionPass()
#     graph = m.graph_for(inputs)
#     print(graph)


# x = torch.randn(4)
# y = torch.randn(4)


# @torch.jit.script
# def foo(a, b):
#     c = a.mul(b)
#     a = c.mul(c)
#     a = c.mul(a)
#     d = c.div(a)
#     return d


# print("original jit ir")
# print(foo.graph_for(x, y))

# jit_res = foo(x, y)

# torch_glow.enableFusionPass()


# @torch.jit.script
# def foo_glow(a, b):
#     return foo(a, b)


# print("glow jit ir")
# print(foo_glow.graph_for(x, y))

# jit_glow_res = foo_glow(x, y)

# print("jit_res")
# print(jit_res)
# print("jit_glow_res")
# print(jit_glow_res)

# assert torch.allclose(jit_res, jit_glow_res)
