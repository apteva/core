# One file handle contract

Incoming files, tool inputs, and tool-produced files use the existing blob handle:

```json
{"_file":true,"ref":"blobref://<id>","filename":"report.pdf","mimeType":"application/pdf","size":12345}
```

`filename` is optional for older tool outputs. The other metadata fields describe
an opaque reference; they do not include file contents. A tool can accept the
reference string or the complete `_file` object. The legacy `_file_ref` wrapper
also remains supported. No separate `fileref://` reference type exists.

To include that same handle in an agent event, wrap it in a content part:

```json
{
  "message": [
    {"type":"text","text":"Process this document."},
    {"type":"file_ref","file_ref":{
      "_file":true,"ref":"blobref://<id>","filename":"report.pdf",
      "mimeType":"application/pdf","size":12345
    }}
  ],
  "event_id":"document-message-1"
}
```

`file_ref` is the event content-part wrapper, not another storage contract. It is
accepted by `POST /event` and thread create/update `events[].message`. Use a
stable `event_id` for durable inbox acceptance and deduplication. Core also
accepts registration metadata from the SDK, but presents only the common handle
fields to the model. Legacy `mime_type` metadata and `apteva-file://` identifiers
remain readable during migration; new producers use `mimeType` and `blobref://`.

## Shared ownership and resolution

The server stores uploaded/tool-generated blobs durably. Apps upload through
`sdk.BlobsClient.StoreBlob`, supplying an authorized agent/thread scope. The
returned `sdk.FileHandle.ContentPart()` can be used directly in an event. Apps
with existing immutable attachment storage can still register an app-owned
source; the server returns the same handle format and resolves that source.

For trusted agent calls, the gateway replaces `_binary` tool outputs with shared
blob handles before they reach core. A later tool call passes the handle back
unchanged. The gateway resolves it only in a declared file input and supplies the
existing `_binary` envelope to the tool. Apps declare those inputs with
`sdk.FileArgumentSchema`; integration binary/multipart inputs already declare
file support. Access is checked against the agent, thread, project and grants.
A handle does not grant access by itself.

Core supplies trusted thread identity from runtime context on signed local MCP
endpoints, separately from model-generated arguments. Core does not fetch or
cache bytes for shared handles. Existing core-local blob stores and unsigned or
streaming MCP transports keep their legacy behavior during migration. They use
the same `blobref://` syntax, but local blobs retain their existing TTL; only
server-backed handles survive process restarts.

## Durable metadata

Reference metadata survives event persistence, session history, retries,
fallbacks, checkpoints, compaction and realtime reconnects. Mixed messages keep
handles when transient images/audio are removed. Normal live context windows
still apply: retained metadata is not an unbounded filesystem or file catalog.

Changes to app-sdk and server must accompany the updated core for shared blob
storage and gateway resolution; upgrading core alone adds only handle input and
persistence support. No new model-facing filesystem or download tool is added.
