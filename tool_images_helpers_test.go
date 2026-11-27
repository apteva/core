package core

// firstToolImageData returns the first image's bytes, or nil.
func firstToolImageData(images []ToolImage) []byte {
	if len(images) == 0 {
		return nil
	}
	return images[0].Data
}
