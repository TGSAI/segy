```{eval-rst}
:tocdepth: 3
```

```{currentModule} segy.schema

```

# Traces

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## Defining a Trace

The [TraceSpec] is a way to define the structure of a seismic trace
as stored in SEG-Y files. It is composed of {ref}`trace-header-specification`
and {ref}`trace-data-specification`. This information is combined using the
[TraceSpec].

The [TraceSpec] has fields for trace header, optional extended trace
header, and trace data definitions. We also provide an optional [offset]
field to define the beginning byte-location of the traces within a binary
file. Most of the time this field gets populated automatically.

A custom trace specification can be built programmatically following a simple
workflow. The same spec can be built from `JSON` as well. Navigate to
[JSON Trace Specification](#json-trace-specification) below for that.

(trace-header-specification)=

### Trace Header Specification

Trace headers are defined using [HeaderSpec]. Each header
field is a [HeaderField]. We have an example workflow here. You
can see more examples in the [Data Types](#data_type) documentation.

We first do the required imports and then define header fields. We don't allow setting
[endianness] for individual fields.

```python

from segy.schema import HeaderField

trace_header_fields = [
    HeaderField(name="inline", byte=189, format="int32"),
    HeaderField(name="crossline", byte=193, format="int32"),
]
```

Then we create [HeaderSpec] for trace headers. We know trace headers must be
240-bytes so we declare it. This will ensure we read/write with correct padding.

```{note}
[Endianness] can be set here but we don't recommend it. By default it will take
the machine endianness. When the [SegyFile](#SegyFile) is initialized it will
automatically set this to the correct value. By default its `None`.
```

```python

from segy.schema import HeaderSpec

trace_header_spec = HeaderSpec(
    fields=trace_header_fields,
    item_size=240,
)
```

(trace-data-specification)=

### Trace Data Specification

Trace data is described using [TraceDataSpec](#TraceDataSpec).
The data is mainly explained by its data [format] and number of [samples].

Continuing our previous example, we build the trace data spec. We assume that samples
are encoded in `ibm32` format. [Endianness] can't be set here, because it is assigned
at [TraceSpec] level.

```python

from segy.schema import TraceDataSpec

trace_data_spec = TraceDataSpec(
    format="ibm32",
    samples=360
)
```

### Trace Specification

Finally, since we have all components, we can create a trace specification.

```python
from segy.schema import TraceSpec

trace_spec = TraceSpec(
    header_spec=trace_header_spec,
    data_spec=trace_data_spec,
    offset=3600  # just an example of possible offset
)
```

```{note}
[Endianness] can be set here but we don't recommend it. By default it will take
the machine endianness. When the [SegyFile](#SegyFile) is initialized it will
automatically set this to the correct value. By default its `None`.
```

If we look at the Numpy data type of the trace, we can see how it
will be decoded from raw bytes:

```python
>>> trace_spec.dtype
dtype([('header', {'names': ['inline', 'crossline'], 'formats': ['<i4', '<i4'], 'offsets': [188, 192], 'itemsize': 240}), ('data', '<u4', (360,))])
```

## JSON Trace Specification

We can define the exact same trace specification above using `JSON`. This can either
be defined as a `string` or can be read from a file. Both will work. Let's write
the `JSON`.

```json
{
  "headerSpec": {
    "fields": [
      {
        "format": "int32",
        "name": "inline",
        "byte": 189
      },
      {
        "format": "int32",
        "name": "crossline",
        "byte": 193
      }
    ],
    "itemSize": 240
  },
  "dataSpec": {
    "format": "ibm32",
    "samples": 360
  },
  "offset": 3600
}
```

Then if we have our `JSON` as a `string` in the variable `json_str`, we can generate
the same specification, with validation of all fields. If there are any errors in the
`JSON`, there will be a validation error raised.

```python
>>> trace_spec_from_json = TraceSpec.model_validate_json(json_str)
>>> trace_spec_from_json == trace_spec
True
```

[tracespec]: #TraceSpec
[offset]: #TraceSpec.offset
[headerspec]: #HeaderSpec
[headerfield]: #HeaderField
[endianness]: #Endianness
[format]: #TraceDataSpec.format
[samples]: #TraceDataSpec.samples

## Reference

```{eval-rst}
.. autopydantic_model:: TraceSpec
```

```{eval-rst}
.. autopydantic_model:: TraceDataSpec
```
