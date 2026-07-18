package visualization

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// authMiddleware returns a handler that enforces bearer-token authentication
// when token is non-empty. Requests without a matching "Authorization: Bearer
// <token>" header receive 401 Unauthorized. When token is empty the handler
// passes through unconditionally. The token comparison is constant-time
// (crypto/subtle) so it does not leak the secret through response timing.
func authMiddleware(token string, next http.Handler) http.Handler {
	if token == "" {
		return next
	}
	want := []byte(token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		got := []byte(strings.TrimPrefix(auth, "Bearer "))
		if !strings.HasPrefix(auth, "Bearer ") || subtle.ConstantTimeCompare(got, want) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="gonn-vis"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// corsMiddleware adds CORS headers when enabled, permitting any origin so
// browser-based dashboards can connect without a proxy. No-op when disabled.
func corsMiddleware(enabled bool, next http.Handler) http.Handler {
	if !enabled {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}
