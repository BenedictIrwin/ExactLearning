# Placeholder

def set_ansatz(key):
    """
    Set the fingerprint and its derivatives in the global space
    """
    if('fingerprint' in globals().keys()): del globals()['fingerprint']
    if('logderivative' in globals().keys()): del globals()['logderivative']
    if('logderivative2' in globals().keys()): del globals()['logderivative2']
    
    # Executes the python in the custom file (i.e. a function)
    # This defines a new function
    with open("./Functions/{}.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())
    with open("./Functions/{}_logderivative.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())
    with open("./Functions/{}_logderivative2.py".format(key),"r") as f: flines = f.readlines()
    exec("".join(flines),globals())