```{eval-rst}
:tocdepth: 3
```

```{currentModule} segy.config

```

# Settings Management

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## `SegyFileSettings` Class

The [SegyFileSettings] is a configuration object for the
[SegyFile] in the environment. It allows you to customize various aspects of
SEG-Y file parsing according to your needs and the specifics of your project.

It is composed of various sub-settings isolated by SEG-Y components and various topics.

- **BINARY**: The [SegyBinaryHeaderSettings] is used for binary header configuration
  while reading a SEG-Y file.
- **ENDIAN**: This setting determines the byte order that is being used in the SEG-Y file.
  The possible options are `"big"` or `"little"` based on [Endianness]. If left as None,
  the system defaults to Big Endian (`"big"`).
- **REVISION**: This setting is used to specify the SEG-Y revision number. If left as
  None, the system will automatically use the revision mentioned in the SEG-Y file.
- **USE_PANDAS**: This setting is a boolean that decides whether to use pandas for
  headers or not. Does not apply to trace data. The trace data is always returned
  as Numpy arrays. The option to use Numpy for headers is currently disabled and will
  be available at a later release (as of March 2024).

## Usage

You initialize an instance of [SegyFileSettings] like any other Python object,
optionally providing initial values for the settings. For example:

```python
from segy.config import SegyBinaryHeaderSettings
from segy.config import SegyFileSettings
from segy.schema import Endianness


# Override extended text header count to zero
binary_header_settings = SegyBinaryHeaderSettings(
    EXTENDED_TEXT_HEADER={"value": 0}
)

settings = SegyFileSettings(
    BINARY=binary_header_settings,
    ENDIAN=Endianness.LITTLE,
    REVISION=1,
)
```

Then this can be passed to [SegyFile] directly.

```python
from segy import SegyFile

file = SegyFile(uri="...", settings=settings)
```

If no settings are provided to [SegyFile], it will take the default values.

```{seealso}
[SegyFileSettings], [SegyFile], [Endianness]
```

## Environment Variables

Environment variables that follow the `SEGY__VARIABLE__SUBVARIABLE` format will be
automatically included in your [SegyFileSettings] instance:

```shell
export SEGY__BINARY__SAMPLES_PER_TRACE__VALUE=1001
export SEGY__BINARY__SAMPLE_INTERVAL__KEY="my_custom_key_in_schema"
export SEGY__ENDIAN="big"
export SEGY__REVISION=0.0
```

The environment variables will override the defaults in the [SegyFileSettings]
configuration, unless user overrides it again within Python.

[endianness]: #Endianness
[segyfilesettings]: #SegyFileSettings
[segyfile]: #SegyFile
[segybinaryheadersettings]: #SegyBinaryHeaderSettings
