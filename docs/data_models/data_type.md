```{eval-rst}
:tocdepth: 3
```

```{currentModule} segy.schema.data_type

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
   StructuredDataTypeDescriptor
   StructuredFieldDescriptor
   Endianness
```

```{eval-rst}
.. autoclass:: ScalarType()
    :members:
    :undoc-members:
    :member-order: bysource
```

```{eval-rst}
.. autopydantic_model:: StructuredDataTypeDescriptor
    :inherited-members: BaseModel
```

```{eval-rst}
.. autopydantic_model:: StructuredFieldDescriptor
    :inherited-members: BaseModel

```

```{eval-rst}
.. autoclass:: Endianness()
    :members:
    :undoc-members:
    :member-order: bysource
```
