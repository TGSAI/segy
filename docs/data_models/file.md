```{eval-rst}
:tocdepth: 3
```

```{currentModule} segy.schema

```

# SEG-Y File

```{article-info}
:author: Altay Sansal
:date: "{sub-ref}`today`"
:read-time: "{sub-ref}`wordcount-minutes` min read"
:class-container: sd-p-0 sd-outline-muted sd-rounded-3 sd-font-weight-light
```

## SEG-Y Spec: A Conceptual Overview

The [SegySpec](#SegySpec) is a structured model used to define the structure and
content of a SEG-Y file. SEG-Y is a standard file format used in the geophysical
industry for recording digital seismic data. In essence, this model serves as a
blueprint for what a SEG-Y file should look like.

This class and its components provide a specified and flexible way to work with
SEG-Y seismic data files programmatically, from defining the file structure and
read/write operations, to customization for specialised use cases.

Conceptually a SEG-Y Revision 0 file looks like this on disk.

```bash
┌──────────────┐  ┌─────────────┐  ┌────────────────────┐        ┌────────────────────┐
│ Textual File │  │ Binary File │  │       Trace 1      │        │       Trace N      │
│ Header 3200B │─►│ Header 400B │─►│ Header 240B + Data │─ ... ─►│ Header 240B + Data │
└──────────────┘  └─────────────┘  └────────────────────┘        └────────────────────┘
```

### Key Components

This spec model consists of several important components. Each of these components
represent a particular section of a SEG-Y file.

#### SEGY-Standard

This attribute, [`segy_standard`](#SegySpec.segy_standard), corresponds
to the specific SEG-Y standard that is being used. SEG-Y files can be of different
revisions or standards, including custom ones.

It must be set to one of the allowed [`SegyStandard`](#SegyStandard) values.

#### Text File Header

The [`text_file_header`](#SegySpec.text_file_header) stores the information
required to parse the textual file header of the SEG-Y file. This includes important
metadata that pertains to the seismic data in human-readable format.

#### Binary File Header

The [`binary_file_header`](#SegySpec.binary_file_header) item talks about
the binary file header of the SEG-Y file. It is a set of structured and important
information about the data in the file, stored in binary format for machines to
read and process quickly and efficiently.

Binary headers are defined as [HeaderSpec](#HeaderSpec) and are built by specifying
header fields in the [HeaderField](#HeaderField) format.

#### Extended Text Header

The [`ext_text_header`](#SegySpec.ext_text_header) is an optional
attribute that provides space for extra information that can't be fit within the
regular text file header. This extended header can be used for additional
human-readable metadata about the data.

```{note}
Extended text headers are were added in SEG-Y Revision 1.0.
```

#### Trace

The [`trace`](#SegySpec.trace) component is a spec definition for both the trace
headers and the associated data. Trace headers contain specific information about
each individual seismic trace in the dataset, and the trace data contains the
actual numerical seismic data.

```{seealso}
[TraceSpec](#TraceSpec)
```

### The Customize Method

The [`customize`](#SegySpec.customize) method is a way for users to tailor an existing
SEG-Y spec to meet their specific requirements. It's an optional tool that provides a
way to update the various parts of the spec including the text header, binary header,
extended text header, trace header and trace data. Note that the SEGY standard
is always set to custom when using this method.

## Reference

```{eval-rst}
.. autopydantic_model:: SegySpec
```

```{eval-rst}
.. autopydantic_model:: TextHeaderSpec
    :inherited-members: BaseModel
```

```{eval-rst}
.. autoclass:: SegyStandard()
    :members:
    :undoc-members:
    :member-order: bysource
```

```{eval-rst}
.. autopydantic_model:: segy.schema.segy.SegyInfo
```
