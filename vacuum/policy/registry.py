# vacuum/policy/registry.py

POLICY_REGISTRY = {}

def register_policy(name):
    def decorator(cls):
        POLICY_REGISTRY[name] = cls
        return cls
    return decorator