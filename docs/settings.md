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

## `SegySettings` Class

The [SegySettings] is a configuration object for the
[SegyFile] in the environment. It allows you to customize various aspects of
SEG-Y file parsing according to your needs and the specifics of your project.

It is composed of various sub-settings isolated by SEG-Y components and various topics.

- **binary**: The [BinaryHeaderSettings] is used for binary header overrides
  when reading a SEG-Y file.
- **endianness**: This setting determines the byte order that is being used in the SEG-Y file.
  The possible options are `"big"` or `"little"` based on [Endianness]. If left as None,
  the system defaults to Big Endian (`"big"`).
- **revision**: This setting is used to specify the SEG-Y revision number. If left as
  None, the system will automatically use the revision mentioned in the SEG-Y file.
- **storage_options**: Provides a hook to pass parameters to storage backend. Like
  credentials, anonymous access, etc.

## Usage

You initialize an instance of [SegySettings] like any other Python object,
optionally providing initial values for the settings. For example:

```python
from segy.config import BinaryHeaderSettings
from segy.config import SegySettings
from segy.schema import Endianness

# Override extended text header count to zero
bin_overrides = BinaryHeaderSettings(extended_text_header=0)

settings = SegySettings(
    binary=bin_overrides,
    endian=Endianness.LITTLE,
    revision=1,
)
```

Then this can be passed to [SegyFile] directly.

```python
from segy import SegyFile

file = SegyFile(uri="...", settings=settings)
```

If no settings are provided to [SegyFile], it will take the default values.

```{seealso}
[SegySettings], [SegyFile], [Endianness]
```

## Environment Variables

Environment variables that follow the `SEGY__VARIABLE__SUBVARIABLE` format will be
automatically included in your [SegySettings] instance:

```shell
export SEGY__BINARY__SAMPLES_PER_TRACE=1001
export SEGY__ENDIANNESS="big"
export SEGY__REVISION=0
```

The environment variables will override the defaults in the [SegySettings]
configuration, unless user overrides it again within Python.

[endianness]: #Endianness
[segysettings]: #SegySettings
[segyfile]: #SegyFile
[segybinaryheadersettings]: #BinaryHeaderSettings
