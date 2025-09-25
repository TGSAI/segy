```{eval-rst}
:tocdepth: 3
```

```{currentModule} segy.schema

```

# Data Types

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## Intro

```{eval-rst}
.. autosummary::
   :nosignatures:

   ScalarType
   HeaderSpec
   HeaderField
   Endianness
```

```{eval-rst}
.. autoclass:: ScalarType()
    :members:
    :undoc-members:
    :member-order: bysource
```

The enumeration also includes ``ScalarType.BYTE`` which represents a single byte
of raw data that is preserved without any endianness conversion. This is useful
for unknown or vendor specific trace header fields that need to maintain bitwise
fidelity.

```{eval-rst}
.. autopydantic_model:: HeaderSpec
    :inherited-members: BaseModel
```

```{eval-rst}
.. autopydantic_model:: HeaderField
    :inherited-members: BaseModel

```

```{eval-rst}
.. autoclass:: Endianness()
    :members:
    :undoc-members:
    :member-order: bysource
```
