package core

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

func validatePublicHTTPURL(raw string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid URL: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("URL scheme must be http or https")
	}
	if u.User != nil || u.Hostname() == "" {
		return nil, fmt.Errorf("URL must contain a host and no userinfo")
	}
	addrs, err := net.DefaultResolver.LookupIPAddr(context.Background(), u.Hostname())
	if err != nil {
		return nil, fmt.Errorf("resolve %q: %w", u.Hostname(), err)
	}
	if len(addrs) == 0 {
		return nil, fmt.Errorf("host %q resolved to no addresses", u.Hostname())
	}
	for _, addr := range addrs {
		if blockedOutboundIP(addr.IP) {
			return nil, fmt.Errorf("URL resolves to blocked address %s", addr.IP)
		}
	}
	return u, nil
}

func blockedOutboundIP(ip net.IP) bool {
	if ip == nil {
		return true
	}
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() || ip.IsUnspecified() || ip.IsMulticast()
}

func newPublicHTTPClient(timeout time.Duration, maxRedirects int) *http.Client {
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(address)
			if err != nil {
				return nil, err
			}
			addrs, err := net.DefaultResolver.LookupIPAddr(ctx, host)
			if err != nil {
				return nil, err
			}
			for _, addr := range addrs {
				if blockedOutboundIP(addr.IP) {
					continue
				}
				return dialer.DialContext(ctx, network, net.JoinHostPort(addr.IP.String(), port))
			}
			return nil, fmt.Errorf("host %q has no permitted addresses", host)
		},
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 20 * time.Second,
	}
	client := &http.Client{Transport: transport, Timeout: timeout}
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) >= maxRedirects {
			return fmt.Errorf("too many redirects")
		}
		_, err := validatePublicHTTPURL(req.URL.String())
		return err
	}
	return client
}

func sameOriginURL(base, target *url.URL) bool {
	if base == nil || target == nil || !strings.EqualFold(base.Scheme, target.Scheme) || !strings.EqualFold(base.Hostname(), target.Hostname()) {
		return false
	}
	port := func(u *url.URL) string {
		if p := u.Port(); p != "" {
			return p
		}
		if u.Scheme == "https" {
			return "443"
		}
		return "80"
	}
	return port(base) == port(target)
}

func runtimeMCPURLAllowed(cfg *Config, raw string) bool {
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Hostname() == "" || u.User != nil {
		return false
	}
	if cfg != nil {
		for _, server := range cfg.GetMCPServers() {
			if server.URL == raw {
				return true
			}
		}
	}
	for _, allowed := range strings.Split(strings.TrimSpace(os.Getenv("APTEVA_MCP_CONNECT_ALLOWLIST")), ",") {
		allowed = strings.TrimSpace(allowed)
		if allowed == "" {
			continue
		}
		if strings.EqualFold(u.Hostname(), allowed) || strings.EqualFold(u.Host, allowed) {
			return true
		}
		if origin, err := url.Parse(allowed); err == nil && sameOriginURL(origin, u) {
			return true
		}
	}
	return false
}
