```{eval-rst}
:tocdepth: 3
```

```{currentModule} segy.schema.trace

```

# Traces

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## Defining a Trace

The [TraceDescriptor] is a way to define the structure of a seismic trace
as stored in SEG-Y files. It is composed of [Trace Header](#trace-header)
and [Trace Data](#trace-data). This information is combined using the
[TraceDescriptor].

The [TraceDescriptor] has fields for trace header, optional extended trace
header, and trace data definitions. We also provide an optional [offset]
field to define the beginning byte-location of the traces within a binary
file. Most of the time this field gets populated automatically.

A custom trace descriptor can be built programmatically following a simple
workflow. The same descriptor can be built from `JSON` as well. Navigate to
[JSON Trace Descriptor](#json-trace-descriptor) below for that.

### Trace Header Descriptor

Trace headers are defined using [StructuredDataTypeDescriptor]. Each header
field is a [StructuredFieldDescriptor]. We have an example workflow here. You
can see more examples in the [Data Types](#data_type) documentation.

We first do the required imports and then define header fields. By default,
endianness is `big`, so we don't have to declare it.

```python
from segy.schema.data_type import StructuredFieldDescriptor

trace_header_fields = [
    StructuredFieldDescriptor(
        name="inline",
        offset=188,
        format="int32",
    ),
    StructuredFieldDescriptor(
        name="crossline",
        offset=192,
        format="int32",
    ),
]
```

Then we create [StructuredDataTypeDescriptor] for trace headers. We know trace
headers must be 240-bytes so we declare it. This will ensure we read/write with
correct padding.

```python
from segy.schema.data_type import StructuredDataTypeDescriptor

trace_header_descriptor = StructuredDataTypeDescriptor(
    fields=trace_header_fields,
    item_size=240,
)
```

### Trace Data Descriptor

Trace data is described using [TraceDataDescriptor](#TraceDataDescriptor).
The data is mainly explained by its data type ([endianness] and [format]),
and number of [samples].

Continuing our previous example, we build the data descriptor. We assume
that samples are encoded in 'ibm32' format and and they are big endian
(again, default).

```python

from segy.schema.trace import TraceDataDescriptor

trace_data_descriptor = TraceDataDescriptor(
    format="ibm32",
    samples=360
)
```

### Trace Descriptor

Finally, since we have all components, we can create a descriptor for of a trace.

```python
from segy.schema.trace import TraceDescriptor

trace_descriptor = TraceDescriptor(
    header_descriptor=trace_header_descriptor,
    data_descriptor=trace_data_descriptor,
    offset=3600  # just an example of possible offset
)
```

If we look at the Numpy data type of the trace, we can see how it
will be decoded from raw bytes:

```python
>>> trace_descriptor.dtype
dtype([('header', {'names': ['inline', 'crossline'], 'formats': ['>i4', '>i4'], 'offsets': [188, 192], 'itemsize': 240}), ('data', '>u4', (360,))])
```

## JSON Trace Descriptor

We can define the exact same trace descriptor above using `JSON`. This can either
be defined as a `string` or can be read from a file. Both will work. Let's write
the `JSON`.

```json
{
  "headerDescriptor": {
    "fields": [
      {
        "format": "int32",
        "name": "inline",
        "offset": 188
      },
      {
        "format": "int32",
        "name": "crossline",
        "offset": 192
      }
    ],
    "itemSize": 240
  },
  "dataDescriptor": {
    "format": "ibm32",
    "samples": 360
  },
  "offset": 3600
}
```

Then if we have our `JSON` as a `string` in the variable `json_str`, we can
generate the same descriptor, with validation of all fields. If there are any
errors in the `JSON`, there will be a validation error raised.

```python
>>> trace_descriptor_from_json = TraceDescriptor.model_validate_json(json_str)
>>> trace_descriptor_from_json == trace_descriptor
True
```

[tracedescriptor]: #TraceDescriptor
[offset]: #TraceDescriptor.offset
[structureddatatypedescriptor]: #StructuredDataTypeDescriptor
[structuredfielddescriptor]: #StructuredFieldDescriptor
[endianness]: #Endianness
[format]: #TraceDataDescriptor.format
[samples]: #TraceDataDescriptor.samples

## Reference

```{eval-rst}
.. autopydantic_model:: TraceDescriptor
```

```{eval-rst}
.. autopydantic_model:: TraceDataDescriptor
```
