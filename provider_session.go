package core

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
)

type providerSessionContextKey struct{}

// Routing identity belongs to a conversation, not a model, request, or cache
// epoch. Instance directory and server distinguish deployments; hashing keeps
// local identifiers out of the header. Auxiliary conversations get their own
// stable purpose suffix. No shared provider state is mutated by workers.
func (t *Thinker) providerSessionContext(ctx context.Context, purpose string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	dir, _ := os.Getwd()
	identity := dir + "\n" + os.Getenv("SERVER_URL") + "\n" + t.promptCacheIdentity() + "\n" + purpose
	sum := sha256.Sum256([]byte(identity))
	return context.WithValue(ctx, providerSessionContextKey{}, "apteva-"+hex.EncodeToString(sum[:16]))
}

func providerSessionFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	id, _ := ctx.Value(providerSessionContextKey{}).(string)
	return id
}
