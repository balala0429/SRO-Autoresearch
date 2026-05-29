OP_TO_ID = {
    # padding (keep both keys for backward compatibility)
    'PAD': 0,
    'pad': 0,

    # variables / constants
    'x': 1,
    'y': 2,
    'const': 3,

    # binary ops
    '+': 4,
    '-': 5,
    '*': 6,
    '/': 7,
    'pow': 8,

    # unary ops
    'sin': 9,
    'cos': 10,
    'exp': 11,
    'log': 12,
    'sqrt': 13,
}