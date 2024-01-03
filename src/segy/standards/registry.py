from segy.schema.segy import SegyDescriptor
from segy.schema.segy import SegyStandard

registry = {}


def register_spec(spec_type: SegyStandard, spec_cls: type[SegyDescriptor]):
    if not issubclass(spec_cls, SegyDescriptor):
        msg = "spec_cls must be a subclass of SegyDescriptor."
        raise ValueError(msg)
    registry[spec_type] = spec_cls


def get_spec(spec_type: SegyStandard):
    spec = registry.get(spec_type)

    if not spec:
        msg = (
            f"Unknown or unsupported SEG-Y spec: {spec_type}. If you "
            f"would like to use {spec_type}, please register it with "
            f"the `SegySpecFactory` using its `register_spec` method."
        )
        raise NotImplementedError(msg)
    return spec()
