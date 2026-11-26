package core

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"net/http"
	"strings"
)

// ToolImage is one image a tool returned with its result. Images stay attached
// to the result that produced them, in the order the tool returned them, so a
// provider can show each one inside (or right after) that tool's result with
// its label.
type ToolImage struct {
	MIMEType string `json:"mime_type,omitempty"`
	Data     []byte `json:"data"`
	// Label tells the model what the image shows ("Frame at 5.0 s"). Core
	// fills a positional label when the tool gives none.
	Label string `json:"label,omitempty"`
	// Retention is the tool's hint for how long the image stays relevant:
	// "latest" (a newer image from the same tool replaces it, like a screen
	// capture) or "keep" (it stays valid until the image budget removes it).
	// Empty means the tool gave no hint.
	Retention string `json:"retention,omitempty"`
}

const (
	ToolImageRetentionLatest = "latest"
	ToolImageRetentionKeep   = "keep"

	// MCP image blocks may carry these _meta keys.
	toolImageLabelMetaKey     = "io.apteva/label"
	toolImageRetentionMetaKey = "io.apteva/retention"

	// A text block directly before an image labels it when it is this short.
	maxToolImageLabelChars = 200

	// Kept tool images share one budget per request. Above the high mark, the
	// oldest retire in one batch down to the low mark (see
	// keptToolImageRetirement). Token figures are estimates, deliberately about
	// twice what providers bill: 48k estimated is roughly 24k real tokens.
	keptToolImagesHighCount  = 16
	keptToolImagesHighTokens = 48_000
	keptToolImagesLowCount   = 8
	keptToolImagesLowTokens  = 24_000
)

// toolImageMetaString reads a string value from an MCP block's _meta.
func toolImageMetaString(meta map[string]json.RawMessage, key string) string {
	raw, ok := meta[key]
	if !ok {
		return ""
	}
	var value string
	if json.Unmarshal(raw, &value) != nil {
		return ""
	}
	return strings.TrimSpace(value)
}

// normalizeToolImageRetention accepts only the two known hints.
func normalizeToolImageRetention(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case ToolImageRetentionLatest:
		return ToolImageRetentionLatest
	case ToolImageRetentionKeep:
		return ToolImageRetentionKeep
	}
	return ""
}

// toolImageLabelFromText uses a short text block as the label of the image
// that follows it. Long or structured text is a result body, not a label.
func toolImageLabelFromText(text string) string {
	text = strings.TrimSpace(text)
	if text == "" || len(text) > maxToolImageLabelChars || strings.ContainsAny(text, "\n{[") {
		return ""
	}
	return text
}

// labelToolImages gives every image a label the model can tie back to the
// call: the tool's own label, or its position in the result.
func labelToolImages(images []ToolImage, toolName string) []ToolImage {
	if len(images) == 0 {
		return images
	}
	out := cloneToolImages(images)
	source := toolName
	if source == "" {
		source = "the tool"
	}
	for i := range out {
		if strings.TrimSpace(out[i].Label) != "" {
			continue
		}
		if len(out) == 1 {
			out[i].Label = "Image from " + source
		} else {
			out[i].Label = fmt.Sprintf("Image %d of %d from %s", i+1, len(out), source)
		}
	}
	return out
}

func cloneToolImages(in []ToolImage) []ToolImage {
	if len(in) == 0 {
		return nil
	}
	out := make([]ToolImage, len(in))
	for i, img := range in {
		out[i] = img
		out[i].Data = append([]byte(nil), img.Data...)
	}
	return out
}

func toolImagesBytes(images []ToolImage) int {
	n := 0
	for _, img := range images {
		n += len(img.Data)
	}
	return n
}

// toolImageMIME returns the declared image type, or sniffs one from the bytes.
func toolImageMIME(img ToolImage) string {
	if mime := strings.ToLower(strings.TrimSpace(img.MIMEType)); strings.HasPrefix(mime, "image/") {
		return mime
	}
	return sniffImageMIME(img.Data)
}

func sniffImageMIME(data []byte) string {
	if _, format, err := image.DecodeConfig(bytes.NewReader(data)); err == nil {
		switch strings.ToLower(format) {
		case "jpeg", "jpg":
			return "image/jpeg"
		case "gif":
			return "image/gif"
		case "png":
			return "image/png"
		case "webp":
			return "image/webp"
		}
	}
	if mime := http.DetectContentType(data); strings.HasPrefix(mime, "image/") {
		return mime
	}
	return "image/png"
}

func toolImageDataURL(img ToolImage) string {
	return "data:" + toolImageMIME(img) + ";base64," + base64.StdEncoding.EncodeToString(img.Data)
}

// toolImageCaption is the text a provider shows right before an image. Tool
// results label every image; only images restored from older history can be
// unlabelled, and those are sent as before, without a caption.
func toolImageCaption(img ToolImage) string {
	if label := strings.TrimSpace(img.Label); label != "" {
		return "[" + label + "]"
	}
	return ""
}

// toolImageParts renders tool images as labelled message parts, for paths
// that cannot attach them to a tool result.
func toolImageParts(images []ToolImage) []ContentPart {
	var parts []ContentPart
	for _, img := range images {
		if caption := toolImageCaption(img); caption != "" {
			parts = append(parts, ContentPart{Type: "text", Text: caption})
		}
		parts = append(parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: toolImageDataURL(img)}})
	}
	return parts
}

// toolImagesToNote returns the images whose removal the model should be told
// about. Replaced screen captures need no note, and without a tool name an
// unhinted image cannot be told apart from one.
func toolImagesToNote(images []ToolImage, toolName string) []ToolImage {
	var out []ToolImage
	for _, img := range images {
		if toolImageIsLatest(img, toolName) {
			continue
		}
		if toolName == "" && img.Retention != ToolImageRetentionKeep {
			continue
		}
		out = append(out, img)
	}
	return out
}

// toolImageIsLatest reports whether a newer image from the same tool replaces
// this one. Tools say so with the "latest" hint. Without a hint, images from
// computer_* tools keep their long-standing screen-capture behaviour.
func toolImageIsLatest(img ToolImage, toolName string) bool {
	switch img.Retention {
	case ToolImageRetentionLatest:
		return true
	case ToolImageRetentionKeep:
		return false
	}
	return isComputerToolName(toolName)
}

// latestToolImageGroup names the set of "latest" images that replace each
// other: all computer_* tools share one screen, other tools have their own.
func latestToolImageGroup(toolName string) string {
	if isComputerToolName(toolName) {
		return "computer"
	}
	return "tool:" + toolName
}

func resultHasLatestImages(result ToolResult, toolName string) bool {
	for _, img := range result.Images {
		if toolImageIsLatest(img, toolName) {
			return true
		}
	}
	return false
}

// splitLatestToolImages separates images a newer result replaces from images
// that stay with the result.
func splitLatestToolImages(images []ToolImage, toolName string) (latest, kept []ToolImage) {
	for _, img := range images {
		if toolImageIsLatest(img, toolName) {
			latest = append(latest, img)
		} else {
			kept = append(kept, img)
		}
	}
	return latest, kept
}

// toolImagesRemovedNote says exactly which images left the context, so the
// model knows it had them and how to see them again.
func toolImagesRemovedNote(removed []ToolImage, toolName, reason string) string {
	if len(removed) == 0 {
		return ""
	}
	source := toolName
	if source == "" {
		source = "this tool"
	}
	labels := make([]string, 0, len(removed))
	total := 0
	for _, img := range removed {
		label := strings.TrimSpace(img.Label)
		if label == "" {
			continue
		}
		if total+len(label) > 600 {
			labels = append(labels, "…")
			break
		}
		total += len(label)
		labels = append(labels, label)
	}
	note := fmt.Sprintf("[%d images returned by %s were removed %s", len(removed), source, reason)
	pronoun := "them"
	if len(removed) == 1 {
		note = fmt.Sprintf("[1 image returned by %s was removed %s", source, reason)
		pronoun = "it"
	}
	if len(labels) > 0 {
		note += ": " + strings.Join(labels, "; ")
	}
	return note + ". You saw " + pronoun + " earlier; call " + source + " again to see " + pronoun + " now.]"
}

func appendToolResultNote(content, note string) string {
	if note == "" {
		return content
	}
	if strings.TrimSpace(content) == "" {
		return note
	}
	return strings.TrimRight(content, "\n") + "\n" + note
}

// UnmarshalJSON reads tool results saved before images became a list: a
// single "image" field becomes the first image.
func (r *ToolResult) UnmarshalJSON(data []byte) error {
	type plain ToolResult
	var aux struct {
		plain
		LegacyImage []byte `json:"image,omitempty"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*r = ToolResult(aux.plain)
	if len(r.Images) == 0 && len(aux.LegacyImage) > 0 {
		r.Images = []ToolImage{{Data: aux.LegacyImage}}
	}
	return nil
}
