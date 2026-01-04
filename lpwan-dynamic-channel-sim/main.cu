// pro_iot_lpwan_sim.cu
// Mô phỏng “nhiều node IoT → nhiều gateway”, kênh động (path loss + shadowing + Rayleigh),
// collision theo kênh + capture effect, heatmap phủ sóng, HUD text (font bitmap tự viết),
// camera pan/zoom, hiển thị beam/pulse chuyên nghiệp hơn.
//
// Build (Linux, g++/clang):
//   nvcc -std=c++17 -O2 -lineinfo -Xcompiler -fopenmp pro_iot_lpwan_sim.cu -o sim $(pkg-config sdl3 --cflags --libs)
//
// Build (Windows MSVC - ví dụ):
//   nvcc -std=c++17 -O2 -lineinfo -Xcompiler "/openmp" pro_iot_lpwan_sim.cu -o sim <SDL3 include/lib flags phù hợp>

#include <omp.h>
#include <cuda_runtime.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t _e = (call);                                   \
    if (_e != cudaSuccess) {                                   \
        SDL_LogError(SDL_LOG_CATEGORY_ERROR,                   \
            "CUDA error %s:%d: %s", __FILE__, __LINE__,        \
            cudaGetErrorString(_e));                           \
        std::exit(1);                                          \
    }                                                          \
} while(0)

static inline float clampf(float x, float a, float b) { return (x < a) ? a : (x > b) ? b : x; }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

// ========================
// Demo parameters (có thể chỉnh)
// ========================
static constexpr int   N_NODES = 2200;
static constexpr int   N_GW    = 3;
static constexpr int   N_CH    = 8;

static constexpr float WORLD_W_M = 1400.0f;
static constexpr float WORLD_H_M = 800.0f;

static constexpr float SIM_FIXED_DT = 1.0f / 60.0f;  // step mô phỏng cố định (60 Hz)
static constexpr float MAX_FRAME_DT = 0.05f;

// Channel model (demo)
static constexpr float Pt_dbm          = 14.0f;
static constexpr float PL0_db          = 40.0f;   // demo
static constexpr float d0_m            = 1.0f;
static constexpr float n_pl            = 2.7f;    // demo
static constexpr float sigma_sh_db     = 6.0f;    // demo
static constexpr float noise_floor_dbm = -120.0f; // demo

// Capture effect (demo): strongest wins if margin >= capture_margin_db
static constexpr float capture_margin_db = 6.0f;

// LoRa-like “SNR requirement by SF” (demo thresholds, không phải thông số chuẩn)
static inline float snr_req_db(int sf) {
    // sf in [7..12]
    switch (sf) {
        case 7:  return -7.5f;
        case 8:  return -10.0f;
        case 9:  return -12.5f;
        case 10: return -15.0f;
        case 11: return -17.5f;
        case 12: return -20.0f;
        default: return -10.0f;
    }
}

// Airtime by SF (demo)
static inline float airtime_s(int sf) {
    switch (sf) {
        case 7:  return 0.060f;
        case 8:  return 0.100f;
        case 9:  return 0.170f;
        case 10: return 0.300f;
        case 11: return 0.550f;
        case 12: return 0.900f;
        default: return 0.150f;
    }
}

struct Gateway {
    float x_m, y_m;
};

enum class LastResult : uint8_t { NONE=0, OK=1, COLLISION=2, SNR_FAIL=3, CAPTURED=4 };

struct NodeCPU {
    float x_m, y_m;
    float vx, vy;

    // traffic
    float next_tx_time_s;
    float tx_end_time_s;
    uint8_t tx_active;
    uint8_t ch;
    uint8_t sf;

    // link selection (best gateway from GPU)
    uint8_t best_gw;

    // dynamic metrics (GPU)
    float rssi_dbm;
    float snr_db;

    // collision state (set each sim tick)
    uint8_t in_collision;  // 1 if group ambiguous/no capture
    uint8_t captured;      // 1 if suppressed by capture
    uint8_t capture_winner;// 1 if winner

    // last finished packet result
    LastResult last_result;
    float last_result_ttl_s; // display time
};

// ========================
// GPU RNG + gaussian
// ========================
__device__ __forceinline__ uint32_t xorshift32(uint32_t &s) {
    s ^= (s << 13);
    s ^= (s >> 17);
    s ^= (s << 5);
    return s;
}
__device__ __forceinline__ float u01(uint32_t &s) {
    uint32_t r = xorshift32(s);
    return (r + 1.0f) * (1.0f / 4294967297.0f); // (0,1)
}
__device__ __forceinline__ float gaussian01(uint32_t &s) {
    float u1 = u01(s);
    float u2 = u01(s);
    float r  = sqrtf(-2.0f * logf(u1));
    float th = 6.28318530718f * u2;
    return r * cosf(th);
}

__device__ __forceinline__ float pathloss_db(float d_m, float PL0, float d0, float n) {
    d_m = fmaxf(d_m, 0.5f);
    return PL0 + 10.0f * n * log10f(d_m / d0);
}

// ========================
// Kernel: compute best RSSI/SNR and best gateway for ALL nodes
// (dynamic: adds shadowing + Rayleigh each call)
// ========================
__global__ void nodes_channel_kernel(
    const float2* __restrict__ pos_m,
    float*        __restrict__ out_rssi_dbm,
    float*        __restrict__ out_snr_db,
    uint8_t*      __restrict__ out_best_gw,
    uint32_t*     __restrict__ rng_state,
    int N,
    const float2* __restrict__ gw_pos_m,
    int n_gw,
    float Pt, float PL0, float d0, float npl,
    float sigma_sh, float noise_floor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint32_t s = rng_state[i];

    float best_rssi = -1e9f;
    int best_idx = 0;

    // 1 shadowing + 1 fading sample per-node per-call (applied per gateway for simplicity)
    float sh_db = sigma_sh * gaussian01(s);
    float g1 = gaussian01(s);
    float g2 = gaussian01(s);
    float hmag = sqrtf(g1*g1 + g2*g2) * 0.70710678118f;
    hmag = fmaxf(hmag, 1e-6f);
    float fad_db = 20.0f * log10f(hmag);

    float2 p = pos_m[i];

    #pragma unroll
    for (int g = 0; g < 8; ++g) { // small unroll upper bound
        if (g >= n_gw) break;
        float2 gw = gw_pos_m[g];
        float dx = p.x - gw.x;
        float dy = p.y - gw.y;
        float d  = sqrtf(dx*dx + dy*dy);
        float PL = pathloss_db(d, PL0, d0, npl);
        float rssi = Pt - (PL + sh_db) + fad_db;

        if (rssi > best_rssi) {
            best_rssi = rssi;
            best_idx = g;
        }
    }

    float snr = best_rssi - noise_floor;

    out_rssi_dbm[i] = best_rssi;
    out_snr_db[i]   = snr;
    out_best_gw[i]  = (uint8_t)best_idx;

    rng_state[i] = s;
}

// ========================
// Kernel: heatmap large-scale RSSI (stable: no shadowing/fading)
// ========================
__global__ void heatmap_kernel(
    float* __restrict__ out_rssi,
    int W, int H,
    float world_w, float world_h,
    const float2* __restrict__ gw_pos_m,
    int n_gw,
    float Pt, float PL0, float d0, float npl
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (idx >= total) return;

    int x = idx % W;
    int y = idx / W;

    float fx = (x + 0.5f) / (float)W;
    float fy = (y + 0.5f) / (float)H;

    float px = fx * world_w;
    float py = fy * world_h;

    float best = -1e9f;

    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g >= n_gw) break;
        float2 gw = gw_pos_m[g];
        float dx = px - gw.x;
        float dy = py - gw.y;
        float d  = sqrtf(dx*dx + dy*dy);
        float PL = pathloss_db(d, PL0, d0, npl);
        float rssi = Pt - PL;
        best = fmaxf(best, rssi);
    }

    out_rssi[idx] = best;
}

// ========================
// Color mapping utilities
// ========================
struct RGBA { uint8_t r,g,b,a; };

static inline RGBA rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a=255) { return RGBA{r,g,b,a}; }

// Smooth “professional-ish” gradient: blue → cyan → green → yellow → red
static inline RGBA rssi_to_color(float rssi_dbm, uint8_t alpha=200) {
    // demo mapping range
    float t = (rssi_dbm - (-125.0f)) / (60.0f); // [-125..-65] => [0..1]
    t = clampf(t, 0.0f, 1.0f);

    // 4 segments
    if (t < 0.25f) { // blue->cyan
        float u = t / 0.25f;
        return rgba(
            (uint8_t)lerpf(30,  40, u),
            (uint8_t)lerpf(70,  220,u),
            (uint8_t)lerpf(255, 255,u),
            alpha
        );
    } else if (t < 0.50f) { // cyan->green
        float u = (t - 0.25f) / 0.25f;
        return rgba(
            (uint8_t)lerpf(40,  60, u),
            (uint8_t)lerpf(220, 255,u),
            (uint8_t)lerpf(255, 80, u),
            alpha
        );
    } else if (t < 0.75f) { // green->yellow
        float u = (t - 0.50f) / 0.25f;
        return rgba(
            (uint8_t)lerpf(60,  255,u),
            (uint8_t)lerpf(255, 240,u),
            (uint8_t)lerpf(80,  60, u),
            alpha
        );
    } else { // yellow->red
        float u = (t - 0.75f) / 0.25f;
        return rgba(
            (uint8_t)lerpf(255, 255,u),
            (uint8_t)lerpf(240, 60, u),
            (uint8_t)lerpf(60,  40, u),
            alpha
        );
    }
}

static inline RGBA result_color(LastResult r) {
    switch (r) {
        case LastResult::OK:        return rgba(90, 255, 130, 255);
        case LastResult::COLLISION: return rgba(255, 70, 70, 255);
        case LastResult::CAPTURED:  return rgba(255, 110, 30, 255);
        case LastResult::SNR_FAIL:  return rgba(255, 190, 60, 255);
        default:                    return rgba(200, 200, 200, 255);
    }
}

// ========================
// Simple camera transform (pan/zoom)
// ========================
struct Camera {
    float cx_m = WORLD_W_M * 0.5f;
    float cy_m = WORLD_H_M * 0.5f;
    float zoom = 0.75f; // pixels per meter (set later based on window)
};

static inline SDL_FPoint world_to_screen(const Camera& cam, float x_m, float y_m, int w_px, int h_px) {
    float sx = (x_m - cam.cx_m) * cam.zoom + (float)w_px * 0.5f;
    float sy = (y_m - cam.cy_m) * cam.zoom + (float)h_px * 0.5f;
    return SDL_FPoint{sx, sy};
}

static inline bool screen_to_world(const Camera& cam, float sx, float sy, int w_px, int h_px, float& out_x, float& out_y) {
    out_x = (sx - (float)w_px * 0.5f) / cam.zoom + cam.cx_m;
    out_y = (sy - (float)h_px * 0.5f) / cam.zoom + cam.cy_m;
    return true;
}

// ========================
// Drawing primitives
// ========================
static void set_color(SDL_Renderer* r, RGBA c) {
    SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a);
}

static void draw_filled_circle(SDL_Renderer* r, float cx, float cy, float radius) {
    int ir = (int)std::lround(radius);
    int icx = (int)std::lround(cx);
    int icy = (int)std::lround(cy);

    for (int dy = -ir; dy <= ir; ++dy) {
        float y = (float)dy;
        float xspan = sqrtf(std::max(0.0f, radius*radius - y*y));
        int x0 = (int)std::lround((float)icx - xspan);
        int x1 = (int)std::lround((float)icx + xspan);
        SDL_RenderLine(r, (float)x0, (float)(icy + dy), (float)x1, (float)(icy + dy));
    }
}

static void draw_circle_outline(SDL_Renderer* r, float cx, float cy, float radius) {
    // Midpoint circle (outline points)
    int x = (int)std::lround(radius);
    int y = 0;
    int err = 0;
    int icx = (int)std::lround(cx);
    int icy = (int)std::lround(cy);

    while (x >= y) {
        SDL_RenderPoint(r, (float)(icx + x), (float)(icy + y));
        SDL_RenderPoint(r, (float)(icx + y), (float)(icy + x));
        SDL_RenderPoint(r, (float)(icx - y), (float)(icy + x));
        SDL_RenderPoint(r, (float)(icx - x), (float)(icy + y));
        SDL_RenderPoint(r, (float)(icx - x), (float)(icy - y));
        SDL_RenderPoint(r, (float)(icx - y), (float)(icy - x));
        SDL_RenderPoint(r, (float)(icx + y), (float)(icy - x));
        SDL_RenderPoint(r, (float)(icx + x), (float)(icy - y));

        if (err <= 0) { y += 1; err += 2*y + 1; }
        if (err > 0)  { x -= 1; err -= 2*x + 1; }
    }
}

static void draw_dashed_beam(SDL_Renderer* r, SDL_FPoint a, SDL_FPoint b, float phase, float dash_px, float gap_px) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float len = sqrtf(dx*dx + dy*dy);
    if (len < 1e-3f) return;
    float ux = dx / len;
    float uy = dy / len;

    float period = dash_px + gap_px;
    float offset = fmodf(phase, period);
    float t = -offset;

    while (t < len) {
        float t0 = std::max(t, 0.0f);
        float t1 = std::min(t + dash_px, len);
        if (t1 > 0.0f && t1 > t0) {
            SDL_RenderLine(r,
                a.x + ux * t0, a.y + uy * t0,
                a.x + ux * t1, a.y + uy * t1
            );
        }
        t += period;
    }
}

// ========================
// Minimal 5x7 bitmap font (uppercase + digits + few symbols)
// Each glyph: 7 rows, 5 bits per row (MSB left).
// ========================
struct Glyph { uint8_t row[7]; };

static Glyph glyph_for(char c) {
    // default: space
    auto SP = Glyph{{0,0,0,0,0,0,0}};

    // Digits 0-9
    switch (c) {
        case '0': return Glyph{{0x1E,0x11,0x13,0x15,0x19,0x11,0x1E}};
        case '1': return Glyph{{0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}};
        case '2': return Glyph{{0x1E,0x01,0x01,0x1E,0x10,0x10,0x1F}};
        case '3': return Glyph{{0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E}};
        case '4': return Glyph{{0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}};
        case '5': return Glyph{{0x1F,0x10,0x10,0x1E,0x01,0x01,0x1E}};
        case '6': return Glyph{{0x0E,0x10,0x10,0x1E,0x11,0x11,0x0E}};
        case '7': return Glyph{{0x1F,0x01,0x02,0x04,0x08,0x08,0x08}};
        case '8': return Glyph{{0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}};
        case '9': return Glyph{{0x0E,0x11,0x11,0x0F,0x01,0x01,0x0E}};
        case ':': return Glyph{{0x00,0x04,0x00,0x00,0x04,0x00,0x00}};
        case '.': return Glyph{{0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}};
        case '-': return Glyph{{0x00,0x00,0x00,0x1F,0x00,0x00,0x00}};
        case '/': return Glyph{{0x01,0x02,0x04,0x08,0x10,0x00,0x00}};
        case '%': return Glyph{{0x19,0x19,0x02,0x04,0x08,0x13,0x13}};
        case ' ': return SP;
        default: break;
    }

    // Uppercase A-Z (subset sufficient for HUD; add more if needed)
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    switch (c) {
        case 'A': return Glyph{{0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}};
        case 'B': return Glyph{{0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}};
        case 'C': return Glyph{{0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}};
        case 'D': return Glyph{{0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}};
        case 'E': return Glyph{{0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}};
        case 'F': return Glyph{{0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}};
        case 'G': return Glyph{{0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}};
        case 'H': return Glyph{{0x11,0x11,0x11,0x1F,0x11,0x11,0x11}};
        case 'I': return Glyph{{0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}};
        case 'K': return Glyph{{0x11,0x12,0x14,0x18,0x14,0x12,0x11}};
        case 'L': return Glyph{{0x10,0x10,0x10,0x10,0x10,0x10,0x1F}};
        case 'M': return Glyph{{0x11,0x1B,0x15,0x11,0x11,0x11,0x11}};
        case 'N': return Glyph{{0x11,0x19,0x15,0x13,0x11,0x11,0x11}};
        case 'O': return Glyph{{0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}};
        case 'P': return Glyph{{0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}};
        case 'R': return Glyph{{0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}};
        case 'S': return Glyph{{0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}};
        case 'T': return Glyph{{0x1F,0x04,0x04,0x04,0x04,0x04,0x04}};
        case 'U': return Glyph{{0x11,0x11,0x11,0x11,0x11,0x11,0x0E}};
        case 'V': return Glyph{{0x11,0x11,0x11,0x11,0x11,0x0A,0x04}};
        case 'W': return Glyph{{0x11,0x11,0x11,0x11,0x15,0x1B,0x11}};
        case 'X': return Glyph{{0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}};
        case 'Y': return Glyph{{0x11,0x11,0x0A,0x04,0x04,0x04,0x04}};
        default: break;
    }
    return SP;
}

static void draw_text(SDL_Renderer* r, float x, float y, const std::string& s, RGBA c, int scale=2) {
    set_color(r, c);
    float cx = x;
    for (char ch : s) {
        Glyph g = glyph_for(ch);
        for (int row = 0; row < 7; ++row) {
            uint8_t bits = g.row[row];
            for (int col = 0; col < 5; ++col) {
                if (bits & (1u << (4 - col))) {
                    SDL_FRect px { cx + col * scale, y + row * scale, (float)scale, (float)scale };
                    SDL_RenderFillRect(r, &px);
                }
            }
        }
        cx += (5 + 1) * scale;
    }
}

// ========================
// Utility: format float (no iostream for speed/size)
// ========================
static std::string f2(float v, int digits=2) {
    char buf[64];
    SDL_snprintf(buf, sizeof(buf), "%.*f", digits, v);
    return std::string(buf);
}
static std::string i2(int v) {
    char buf[64];
    SDL_snprintf(buf, sizeof(buf), "%d", v);
    return std::string(buf);
}

// ========================
// Main
// ========================
int main(int argc, char** argv) {
    (void)argc; (void)argv;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "SDL_Init failed: %s", SDL_GetError());
        return 1;
    }

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;

    const int WIN_W0 = 1280, WIN_H0 = 720;
    if (!SDL_CreateWindowAndRenderer("Pro IoT/LPWAN Channel Simulator", WIN_W0, WIN_H0, SDL_WINDOW_RESIZABLE, &window, &renderer)) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "SDL_CreateWindowAndRenderer failed: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    // camera init based on window
    Camera cam;
    cam.zoom = std::min((float)WIN_W0 / WORLD_W_M, (float)WIN_H0 / WORLD_H_M) * 0.92f;

    // Gateways layout (demo)
    Gateway gws[N_GW] = {
        { WORLD_W_M * 0.25f, WORLD_H_M * 0.35f },
        { WORLD_W_M * 0.75f, WORLD_H_M * 0.35f },
        { WORLD_W_M * 0.50f, WORLD_H_M * 0.70f },
    };

    // Host node state
    std::vector<NodeCPU> nodes(N_NODES);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ux(0.0f, WORLD_W_M);
    std::uniform_real_distribution<float> uy(0.0f, WORLD_H_M);
    std::uniform_real_distribution<float> uv(-10.0f, 10.0f);
    std::uniform_real_distribution<float> uu(0.0f, 1.0f);

    // Traffic model: Poisson per node with mean interval
    float mean_interval_s = 4.0f;        // demo: trung bình 1 gói / 4s / node
    float sim_speed = 1.0f;              // speed multiplier

    for (int i = 0; i < N_NODES; ++i) {
        nodes[i].x_m = ux(rng);
        nodes[i].y_m = uy(rng);
        nodes[i].vx  = uv(rng);
        nodes[i].vy  = uv(rng);

        float u = std::max(1e-6f, uu(rng));
        nodes[i].next_tx_time_s = -std::log(u) * mean_interval_s;
        nodes[i].tx_end_time_s  = 0.0f;
        nodes[i].tx_active      = 0;
        nodes[i].ch             = 0;
        nodes[i].sf             = 7;
        nodes[i].best_gw        = 0;
        nodes[i].rssi_dbm       = -140.0f;
        nodes[i].snr_db         = -999.0f;
        nodes[i].in_collision   = 0;
        nodes[i].captured       = 0;
        nodes[i].capture_winner = 0;
        nodes[i].last_result    = LastResult::NONE;
        nodes[i].last_result_ttl_s = 0.0f;
    }

    // ================= GPU buffers =================
    float2*   d_pos = nullptr;
    float*    d_rssi = nullptr;
    float*    d_snr  = nullptr;
    uint8_t*  d_best_gw = nullptr;
    uint32_t* d_rng = nullptr;

    float2*   d_gw_pos = nullptr;

    // Heatmap buffers
    const int HM_W = 260;
    const int HM_H = 150;
    float* d_heat = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(float2) * N_NODES));
    CUDA_CHECK(cudaMalloc(&d_rssi, sizeof(float) * N_NODES));
    CUDA_CHECK(cudaMalloc(&d_snr, sizeof(float) * N_NODES));
    CUDA_CHECK(cudaMalloc(&d_best_gw, sizeof(uint8_t) * N_NODES));
    CUDA_CHECK(cudaMalloc(&d_rng, sizeof(uint32_t) * N_NODES));

    CUDA_CHECK(cudaMalloc(&d_gw_pos, sizeof(float2) * N_GW));
    CUDA_CHECK(cudaMalloc(&d_heat, sizeof(float) * HM_W * HM_H));

    // init gateway positions
    std::vector<float2> h_gw(N_GW);
    for (int g = 0; g < N_GW; ++g) h_gw[g] = make_float2(gws[g].x_m, gws[g].y_m);
    CUDA_CHECK(cudaMemcpy(d_gw_pos, h_gw.data(), sizeof(float2)*N_GW, cudaMemcpyHostToDevice));

    // init RNG states
    std::vector<uint32_t> h_rng(N_NODES);
    for (int i = 0; i < N_NODES; ++i) {
        uint32_t seed = (uint32_t)(0xA341316Cu ^ (i * 2654435761u));
        h_rng[i] = seed ? seed : 1u;
    }
    CUDA_CHECK(cudaMemcpy(d_rng, h_rng.data(), sizeof(uint32_t)*N_NODES, cudaMemcpyHostToDevice));

    // pinned host buffers for faster transfers
    float2*  h_pos_pinned = nullptr;
    float*   h_rssi_pinned = nullptr;
    float*   h_snr_pinned  = nullptr;
    uint8_t* h_bestgw_pinned = nullptr;

    CUDA_CHECK(cudaHostAlloc(&h_pos_pinned,    sizeof(float2) * N_NODES, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_rssi_pinned,   sizeof(float)  * N_NODES, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_snr_pinned,    sizeof(float)  * N_NODES, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_bestgw_pinned, sizeof(uint8_t)* N_NODES, cudaHostAllocDefault));

    // heatmap host
    std::vector<float> h_heat(HM_W * HM_H);
    std::vector<uint32_t> heat_rgba(HM_W * HM_H);

    // SDL texture for heatmap
    SDL_Texture* heat_tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, HM_W, HM_H);
    SDL_SetTextureBlendMode(heat_tex, SDL_BLENDMODE_BLEND);

    // CUDA streams
    cudaStream_t stream_nodes, stream_heat;
    CUDA_CHECK(cudaStreamCreate(&stream_nodes));
    CUDA_CHECK(cudaStreamCreate(&stream_heat));

    // heatmap update pacing
    float heat_timer = 0.0f;
    const float HEAT_UPDATE_INTERVAL = 0.25f; // update 4 Hz for stability

    // toggles
    bool running = true;
    bool paused  = false;
    bool show_heat = true;
    bool show_grid = true;
    bool show_links = true;
    bool show_rings = true;

    // camera input state
    bool panning = false;
    float pan_last_x = 0.0f, pan_last_y = 0.0f;

    // stats
    uint64_t total_tx = 0;
    uint64_t total_ok = 0;
    uint64_t total_col = 0;
    uint64_t total_snr_fail = 0;
    uint64_t total_captured = 0;

    // FPS
    uint64_t last_ticks = SDL_GetTicks();
    uint64_t fps_ticks_acc = 0;
    int fps_frames_acc = 0;
    float fps_est = 0.0f;

    // simulation time
    double sim_time = 0.0;
    double accum = 0.0;

    // scratch: active indices
    std::vector<int> active_idx;
    active_idx.reserve(N_NODES);

    // scratch: grouping by gateway+channel
    // store list indices for each bucket
    std::vector<std::vector<int>> buckets(N_GW * N_CH);

    auto bucket_id = [&](int gw, int ch) { return gw * N_CH + ch; };

    while (running) {
        // ===== events =====
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (ev.type == SDL_EVENT_KEY_DOWN) {
                SDL_Scancode sc = ev.key.scancode;
                if (sc == SDL_SCANCODE_ESCAPE) running = false;
                else if (sc == SDL_SCANCODE_SPACE) paused = !paused;
                else if (sc == SDL_SCANCODE_H) show_heat = !show_heat;
                else if (sc == SDL_SCANCODE_G) show_grid = !show_grid;
                else if (sc == SDL_SCANCODE_L) show_links = !show_links;
                else if (sc == SDL_SCANCODE_C) show_rings = !show_rings;
                else if (sc == SDL_SCANCODE_R) {
                    total_tx = total_ok = total_col = total_snr_fail = total_captured = 0;
                    sim_time = 0.0;
                    accum = 0.0;
                    for (int i = 0; i < N_NODES; ++i) {
                        float u = std::max(1e-6f, uu(rng));
                        nodes[i].next_tx_time_s = -std::log(u) * mean_interval_s;
                        nodes[i].tx_active = 0;
                        nodes[i].last_result = LastResult::NONE;
                        nodes[i].last_result_ttl_s = 0.0f;
                    }
                }
                else if (sc == SDL_SCANCODE_EQUALS || sc == SDL_SCANCODE_KP_PLUS) {
                    sim_speed = std::min(8.0f, sim_speed * 1.25f);
                }
                else if (sc == SDL_SCANCODE_MINUS || sc == SDL_SCANCODE_KP_MINUS) {
                    sim_speed = std::max(0.25f, sim_speed / 1.25f);
                }
                else if (sc == SDL_SCANCODE_UP) {
                    mean_interval_s = std::max(0.5f, mean_interval_s * 0.85f);
                }
                else if (sc == SDL_SCANCODE_DOWN) {
                    mean_interval_s = std::min(30.0f, mean_interval_s * 1.15f);
                }
            } else if (ev.type == SDL_EVENT_MOUSE_WHEEL) {
                // zoom at mouse position
                int w_px=0,h_px=0; SDL_GetWindowSizeInPixels(window, &w_px, &h_px);
                float mx = 0.0f, my = 0.0f;
                SDL_GetMouseState(&mx, &my);

                float wx0, wy0;
                screen_to_world(cam, mx, my, w_px, h_px, wx0, wy0);

                float z = cam.zoom;
                float factor = (ev.wheel.y > 0) ? 1.12f : 0.89f;
                cam.zoom = clampf(z * factor, 0.10f, 3.50f);

                float wx1, wy1;
                screen_to_world(cam, mx, my, w_px, h_px, wx1, wy1);

                // keep the world point under cursor fixed
                cam.cx_m += (wx0 - wx1);
                cam.cy_m += (wy0 - wy1);
            } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                if (ev.button.button == SDL_BUTTON_RIGHT) {
                    panning = true;
                    pan_last_x = ev.button.x;
                    pan_last_y = ev.button.y;
                }
            } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                if (ev.button.button == SDL_BUTTON_RIGHT) {
                    panning = false;
                }
            } else if (ev.type == SDL_EVENT_MOUSE_MOTION) {
                if (panning) {
                    int w_px=0,h_px=0; SDL_GetWindowSizeInPixels(window, &w_px, &h_px);
                    float dx = ev.motion.x - pan_last_x;
                    float dy = ev.motion.y - pan_last_y;
                    pan_last_x = ev.motion.x;
                    pan_last_y = ev.motion.y;

                    // pan camera opposite mouse drag
                    cam.cx_m -= dx / cam.zoom;
                    cam.cy_m -= dy / cam.zoom;

                    cam.cx_m = clampf(cam.cx_m, 0.0f, WORLD_W_M);
                    cam.cy_m = clampf(cam.cy_m, 0.0f, WORLD_H_M);
                }
            }
        }

        // ===== time =====
        uint64_t now_ticks = SDL_GetTicks();
        float frame_dt = (float)((now_ticks - last_ticks) * 0.001);
        last_ticks = now_ticks;
        frame_dt = clampf(frame_dt, 0.0f, MAX_FRAME_DT);

        // fps estimate
        fps_ticks_acc += (uint64_t)(frame_dt * 1000.0f);
        fps_frames_acc++;
        if (fps_ticks_acc >= 500) {
            fps_est = (fps_frames_acc * 1000.0f) / (float)fps_ticks_acc;
            fps_ticks_acc = 0;
            fps_frames_acc = 0;
        }

        // ===== simulation step(s) =====
        if (!paused) accum += frame_dt * sim_speed;

        while (!paused && accum >= SIM_FIXED_DT) {
            float dt = SIM_FIXED_DT;
            accum -= dt;
            sim_time += dt;

            // mobility update (OpenMP)
            #pragma omp parallel for
            for (int i = 0; i < N_NODES; ++i) {
                nodes[i].x_m += nodes[i].vx * dt;
                nodes[i].y_m += nodes[i].vy * dt;

                // bounce
                if (nodes[i].x_m < 0.0f)       { nodes[i].x_m = 0.0f;       nodes[i].vx = -nodes[i].vx; }
                if (nodes[i].x_m > WORLD_W_M)  { nodes[i].x_m = WORLD_W_M;  nodes[i].vx = -nodes[i].vx; }
                if (nodes[i].y_m < 0.0f)       { nodes[i].y_m = 0.0f;       nodes[i].vy = -nodes[i].vy; }
                if (nodes[i].y_m > WORLD_H_M)  { nodes[i].y_m = WORLD_H_M;  nodes[i].vy = -nodes[i].vy; }

                // decay last-result highlight
                if (nodes[i].last_result_ttl_s > 0.0f) {
                    nodes[i].last_result_ttl_s = std::max(0.0f, nodes[i].last_result_ttl_s - dt);
                    if (nodes[i].last_result_ttl_s == 0.0f) nodes[i].last_result = LastResult::NONE;
                }
            }

            // schedule starts / ends (CPU)
            active_idx.clear();
            for (int b = 0; b < (int)buckets.size(); ++b) buckets[b].clear();

            for (int i = 0; i < N_NODES; ++i) {
                // end tx?
                if (nodes[i].tx_active && (float)sim_time >= nodes[i].tx_end_time_s) {
                    nodes[i].tx_active = 0;

                    // decide final result (based on flags that were set during last evaluation tick)
                    if (nodes[i].captured) {
                        nodes[i].last_result = LastResult::CAPTURED;
                        total_captured++;
                    } else if (nodes[i].in_collision) {
                        nodes[i].last_result = LastResult::COLLISION;
                        total_col++;
                    } else {
                        // no collision -> SNR threshold by SF
                        if (nodes[i].snr_db >= snr_req_db(nodes[i].sf)) {
                            nodes[i].last_result = LastResult::OK;
                            total_ok++;
                        } else {
                            nodes[i].last_result = LastResult::SNR_FAIL;
                            total_snr_fail++;
                        }
                    }
                    nodes[i].last_result_ttl_s = 1.0f; // show for 1s
                }

                // start tx?
                if (!nodes[i].tx_active && (float)sim_time >= nodes[i].next_tx_time_s) {
                    // choose channel
                    uint32_t s = (uint32_t)(i * 747796405u) ^ (uint32_t)((uint32_t)(sim_time*1000.0) * 2891336453u);
                    uint8_t ch = (uint8_t)(s % N_CH);
                    nodes[i].ch = ch;

                    // choose SF from distance to nearest gateway (CPU quick)
                    float best_d2 = 1e30f;
                    int best_g = 0;
                    for (int g = 0; g < N_GW; ++g) {
                        float dx = nodes[i].x_m - gws[g].x_m;
                        float dy = nodes[i].y_m - gws[g].y_m;
                        float d2 = dx*dx + dy*dy;
                        if (d2 < best_d2) { best_d2 = d2; best_g = g; }
                    }
                    float d = sqrtf(best_d2);
                    int sf = 7;
                    if      (d > 950.0f) sf = 12;
                    else if (d > 800.0f) sf = 11;
                    else if (d > 650.0f) sf = 10;
                    else if (d > 520.0f) sf = 9;
                    else if (d > 400.0f) sf = 8;
                    else                 sf = 7;
                    nodes[i].sf = (uint8_t)sf;

                    // start tx
                    nodes[i].tx_active = 1;
                    nodes[i].tx_end_time_s = (float)sim_time + airtime_s(sf);

                    // schedule next tx (Poisson)
                    float u = std::max(1e-6f, uu(rng));
                    nodes[i].next_tx_time_s = (float)sim_time + (-std::log(u) * mean_interval_s);

                    // count stats
                    total_tx++;

                    // reset collision flags for this transmission
                    nodes[i].in_collision = 0;
                    nodes[i].captured = 0;
                    nodes[i].capture_winner = 0;
                }

                // build active list
                if (nodes[i].tx_active) {
                    active_idx.push_back(i);
                }
            }

            // ===== GPU: compute rssi/snr/best gw for all nodes (used for coloring + collision buckets) =====
            // pack positions into pinned buffer
            #pragma omp parallel for
            for (int i = 0; i < N_NODES; ++i) {
                h_pos_pinned[i] = make_float2(nodes[i].x_m, nodes[i].y_m);
            }

            CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos_pinned, sizeof(float2)*N_NODES, cudaMemcpyHostToDevice, stream_nodes));

            int block = 256;
            int grid  = (N_NODES + block - 1) / block;
            nodes_channel_kernel<<<grid, block, 0, stream_nodes>>>(
                d_pos, d_rssi, d_snr, d_best_gw, d_rng,
                N_NODES, d_gw_pos, N_GW,
                Pt_dbm, PL0_db, d0_m, n_pl,
                sigma_sh_db, noise_floor_dbm
            );
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(h_rssi_pinned, d_rssi, sizeof(float)*N_NODES, cudaMemcpyDeviceToHost, stream_nodes));
            CUDA_CHECK(cudaMemcpyAsync(h_snr_pinned,  d_snr,  sizeof(float)*N_NODES, cudaMemcpyDeviceToHost, stream_nodes));
            CUDA_CHECK(cudaMemcpyAsync(h_bestgw_pinned, d_best_gw, sizeof(uint8_t)*N_NODES, cudaMemcpyDeviceToHost, stream_nodes));

            CUDA_CHECK(cudaStreamSynchronize(stream_nodes));

            // apply GPU results back to nodes
            #pragma omp parallel for
            for (int i = 0; i < N_NODES; ++i) {
                nodes[i].rssi_dbm = h_rssi_pinned[i];
                nodes[i].snr_db   = h_snr_pinned[i];
                nodes[i].best_gw  = h_bestgw_pinned[i];

                // clear per-tick collision flags (will be recomputed)
                nodes[i].in_collision = 0;
                nodes[i].captured = 0;
                nodes[i].capture_winner = 0;
            }

            // ===== Collision/capture evaluation for ACTIVE transmissions =====
            // Bucket active tx by (best_gw, ch)
            for (int idx : active_idx) {
                int gw = nodes[idx].best_gw;
                int ch = nodes[idx].ch;
                buckets[bucket_id(gw, ch)].push_back(idx);
            }

            for (int gw = 0; gw < N_GW; ++gw) {
                for (int ch = 0; ch < N_CH; ++ch) {
                    auto &list = buckets[bucket_id(gw, ch)];
                    if (list.size() <= 1) continue;

                    // find top1 & top2 RSSI
                    int top1 = -1, top2 = -1;
                    float r1 = -1e9f, r2 = -1e9f;

                    for (int idx : list) {
                        float rssi = nodes[idx].rssi_dbm;
                        if (rssi > r1) {
                            r2 = r1; top2 = top1;
                            r1 = rssi; top1 = idx;
                        } else if (rssi > r2) {
                            r2 = rssi; top2 = idx;
                        }
                    }

                    if (top1 >= 0 && top2 >= 0 && (r1 - r2) >= capture_margin_db) {
                        // capture: top1 wins, others captured
                        nodes[top1].capture_winner = 1;
                        for (int idx : list) {
                            if (idx == top1) continue;
                            nodes[idx].captured = 1;
                        }
                    } else {
                        // ambiguous: all collide
                        for (int idx : list) nodes[idx].in_collision = 1;
                    }
                }
            }

            // ===== Heatmap update (slower rate) =====
            heat_timer += dt;
            if (show_heat && heat_timer >= HEAT_UPDATE_INTERVAL) {
                heat_timer = 0.0f;

                int total = HM_W * HM_H;
                int grid2 = (total + block - 1) / block;

                heatmap_kernel<<<grid2, block, 0, stream_heat>>>(
                    d_heat, HM_W, HM_H,
                    WORLD_W_M, WORLD_H_M,
                    d_gw_pos, N_GW,
                    Pt_dbm, PL0_db, d0_m, n_pl
                );
                CUDA_CHECK(cudaGetLastError());

                CUDA_CHECK(cudaMemcpyAsync(h_heat.data(), d_heat, sizeof(float)*total, cudaMemcpyDeviceToHost, stream_heat));
                CUDA_CHECK(cudaStreamSynchronize(stream_heat));

                // convert to RGBA pixels
                for (int i = 0; i < total; ++i) {
                    RGBA c = rssi_to_color(h_heat[i], 110);
                    heat_rgba[i] = (uint32_t)c.r | ((uint32_t)c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)c.a << 24);
                }

                // upload to SDL texture
                void* pixels = nullptr;
                int pitch = 0;
                if (SDL_LockTexture(heat_tex, nullptr, &pixels, &pitch)) {
                    uint8_t* dst = (uint8_t*)pixels;
                    for (int y = 0; y < HM_H; ++y) {
                        std::memcpy(dst + y * pitch, &heat_rgba[y * HM_W], HM_W * sizeof(uint32_t));
                    }
                    SDL_UnlockTexture(heat_tex);
                }
            }
        }

        // ===== render =====
        int w_px=0,h_px=0;
        SDL_GetWindowSizeInPixels(window, &w_px, &h_px);

        // background
        SDL_SetRenderDrawColor(renderer, 10, 10, 14, 255);
        SDL_RenderClear(renderer);

        // heatmap (world rect -> screen rect)
        if (show_heat) {
            SDL_FPoint p0 = world_to_screen(cam, 0.0f, 0.0f, w_px, h_px);
            SDL_FPoint p1 = world_to_screen(cam, WORLD_W_M, WORLD_H_M, w_px, h_px);

            SDL_FRect dst;
            dst.x = std::min(p0.x, p1.x);
            dst.y = std::min(p0.y, p1.y);
            dst.w = std::fabs(p1.x - p0.x);
            dst.h = std::fabs(p1.y - p0.y);

            SDL_RenderTexture(renderer, heat_tex, nullptr, &dst);
        }

        // grid
        if (show_grid) {
            set_color(renderer, rgba(255,255,255,18));
            const float step_m = 100.0f;

            // vertical lines
            for (float x = 0.0f; x <= WORLD_W_M + 0.5f; x += step_m) {
                SDL_FPoint a = world_to_screen(cam, x, 0.0f, w_px, h_px);
                SDL_FPoint b = world_to_screen(cam, x, WORLD_H_M, w_px, h_px);
                SDL_RenderLine(renderer, a.x, a.y, b.x, b.y);
            }
            // horizontal lines
            for (float y = 0.0f; y <= WORLD_H_M + 0.5f; y += step_m) {
                SDL_FPoint a = world_to_screen(cam, 0.0f, y, w_px, h_px);
                SDL_FPoint b = world_to_screen(cam, WORLD_W_M, y, w_px, h_px);
                SDL_RenderLine(renderer, a.x, a.y, b.x, b.y);
            }
        }

        // gateways + rings
        for (int g = 0; g < N_GW; ++g) {
            SDL_FPoint pg = world_to_screen(cam, gws[g].x_m, gws[g].y_m, w_px, h_px);

            // rings
            if (show_rings) {
                set_color(renderer, rgba(255, 230, 120, 40));
                for (int k = 1; k <= 4; ++k) {
                    float r_m = 200.0f * k;
                    float r_px = r_m * cam.zoom;
                    draw_circle_outline(renderer, pg.x, pg.y, r_px);
                }
            }

            // core + border
            set_color(renderer, rgba(255, 230, 120, 255));
            draw_filled_circle(renderer, pg.x, pg.y, std::max(4.0f, 6.0f * cam.zoom * 0.6f));

            set_color(renderer, rgba(255, 255, 255, 130));
            draw_circle_outline(renderer, pg.x, pg.y, std::max(6.0f, 10.0f * cam.zoom * 0.6f));
        }

        // beams + nodes
        float beam_phase = (float)fmod(sim_time * 180.0, 100000.0);

        // draw beams first (so nodes overlay on top)
        if (show_links) {
            for (int i = 0; i < N_NODES; ++i) {
                if (!nodes[i].tx_active) continue;

                int gw = nodes[i].best_gw;
                SDL_FPoint a = world_to_screen(cam, nodes[i].x_m, nodes[i].y_m, w_px, h_px);
                SDL_FPoint b = world_to_screen(cam, gws[gw].x_m, gws[gw].y_m, w_px, h_px);

                // choose beam color by collision state
                RGBA c = rgba(90, 200, 255, 180); // transmitting
                if (nodes[i].captured) c = rgba(255, 120, 40, 200);
                else if (nodes[i].in_collision) c = rgba(255, 70, 70, 200);
                else if (nodes[i].capture_winner) c = rgba(90, 255, 130, 200);

                set_color(renderer, c);

                // dashed moving beam
                draw_dashed_beam(renderer, a, b, beam_phase, 10.0f, 8.0f);
            }
        }

        // nodes
        for (int i = 0; i < N_NODES; ++i) {
            SDL_FPoint p = world_to_screen(cam, nodes[i].x_m, nodes[i].y_m, w_px, h_px);

            // base color from RSSI
            RGBA base = rssi_to_color(nodes[i].rssi_dbm, 230);

            // if tx active, brighten
            if (nodes[i].tx_active) {
                base = rgba(
                    (uint8_t)std::min(255, base.r + 30),
                    (uint8_t)std::min(255, base.g + 30),
                    (uint8_t)std::min(255, base.b + 30),
                    255
                );
            }

            set_color(renderer, base);

            float r_px = std::max(1.5f, 2.2f * cam.zoom * 0.55f);
            draw_filled_circle(renderer, p.x, p.y, r_px);

            // tx pulse ring
            if (nodes[i].tx_active) {
                float t = (float)fmod(sim_time * 3.0, 1.0);
                float pr = (2.0f + 10.0f * t) * (0.6f + 0.4f * cam.zoom);
                set_color(renderer, rgba(255,255,255, (uint8_t)lerpf(80, 0, t)));
                draw_circle_outline(renderer, p.x, p.y, pr);
            }

            // last result badge
            if (nodes[i].last_result != LastResult::NONE && nodes[i].last_result_ttl_s > 0.0f) {
                float t = nodes[i].last_result_ttl_s; // 0..1
                RGBA rc = result_color(nodes[i].last_result);
                rc.a = (uint8_t)lerpf(0, rc.a, clampf(t, 0.0f, 1.0f));
                set_color(renderer, rc);
                draw_circle_outline(renderer, p.x, p.y, r_px + 3.0f);
            }
        }

        // HUD panel
        {
            // panel background
            SDL_FRect panel { 16.0f, 16.0f, 420.0f, 140.0f };
            set_color(renderer, rgba(0,0,0,140));
            SDL_RenderFillRect(renderer, &panel);
            set_color(renderer, rgba(255,255,255,40));
            SDL_RenderRect(renderer, &panel);

            // metrics
            double pdr = (total_tx > 0) ? (double)total_ok / (double)total_tx : 0.0;

            int active_count = 0;
            for (int i=0;i<N_NODES;++i) active_count += (nodes[i].tx_active ? 1 : 0);

            std::string line1 = "PRO IOT/LPWAN CHANNEL SIM";
            std::string line2 = "N:" + i2(N_NODES) + "  GW:" + i2(N_GW) + "  CH:" + i2(N_CH);
            std::string line3 = "SIM T:" + f2((float)sim_time,1) + "s  SPEED:" + f2(sim_speed,2) + "x  FPS:" + f2(fps_est,1);
            std::string line4 = "MEAN INT:" + f2(mean_interval_s,2) + "s  ACTIVE TX:" + i2(active_count);
            std::string line5 = "TX:" + i2((int)total_tx) + "  OK:" + i2((int)total_ok) + "  PDR:" + f2((float)pdr,3);
            std::string line6 = "COL:" + i2((int)total_col) + "  CAP:" + i2((int)total_captured) + "  SNRFAIL:" + i2((int)total_snr_fail);

            draw_text(renderer, 28.0f,  28.0f, line1, rgba(255,255,255,220), 2);
            draw_text(renderer, 28.0f,  48.0f, line2, rgba(210,210,210,220), 2);
            draw_text(renderer, 28.0f,  68.0f, line3, rgba(210,210,210,220), 2);
            draw_text(renderer, 28.0f,  88.0f, line4, rgba(210,210,210,220), 2);
            draw_text(renderer, 28.0f, 108.0f, line5, rgba(210,210,210,220), 2);
            draw_text(renderer, 28.0f, 128.0f, line6, rgba(210,210,210,220), 2);

            // controls hint
            std::string hint = "H:HEAT  G:GRID  L:LINK  C:RING  SPACE:PAUSE  +/-:SPEED  UP/DN:RATE  R:RESET";
            draw_text(renderer, 16.0f, (float)h_px - 24.0f, hint, rgba(255,255,255,140), 1);
        }

        // legend bar (RSSI)
        {
            float x0 = 16.0f, y0 = (float)h_px - 54.0f;
            float w = 260.0f, h = 10.0f;
            for (int i = 0; i < (int)w; ++i) {
                float t = i / (w - 1.0f);
                float rssi = lerpf(-125.0f, -65.0f, t);
                RGBA c = rssi_to_color(rssi, 220);
                set_color(renderer, c);
                SDL_FRect r { x0 + i, y0, 1.0f, h };
                SDL_RenderFillRect(renderer, &r);
            }
            set_color(renderer, rgba(255,255,255,60));
            SDL_FRect box { x0, y0, w, h };
            SDL_RenderRect(renderer, &box);

            draw_text(renderer, x0, y0 - 14.0f, "RSSI DBM", rgba(255,255,255,130), 1);
            draw_text(renderer, x0, y0 + 14.0f, "-125", rgba(255,255,255,130), 1);
            draw_text(renderer, x0 + w - 30.0f, y0 + 14.0f, "-65", rgba(255,255,255,130), 1);
        }

        SDL_RenderPresent(renderer);

        // window title (short)
        {
            double pdr = (total_tx > 0) ? (double)total_ok / (double)total_tx : 0.0;
            char title[256];
            SDL_snprintf(title, sizeof(title),
                "Pro IoT/LPWAN Sim | N=%d GW=%d CH=%d | TX=%llu OK=%llu PDR=%.3f | FPS=%.1f",
                N_NODES, N_GW, N_CH,
                (unsigned long long)total_tx,
                (unsigned long long)total_ok,
                pdr,
                fps_est
            );
            SDL_SetWindowTitle(window, title);
        }

        SDL_Delay(1);
    }

    // cleanup
    CUDA_CHECK(cudaStreamDestroy(stream_nodes));
    CUDA_CHECK(cudaStreamDestroy(stream_heat));

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_rssi));
    CUDA_CHECK(cudaFree(d_snr));
    CUDA_CHECK(cudaFree(d_best_gw));
    CUDA_CHECK(cudaFree(d_rng));
    CUDA_CHECK(cudaFree(d_gw_pos));
    CUDA_CHECK(cudaFree(d_heat));

    CUDA_CHECK(cudaFreeHost(h_pos_pinned));
    CUDA_CHECK(cudaFreeHost(h_rssi_pinned));
    CUDA_CHECK(cudaFreeHost(h_snr_pinned));
    CUDA_CHECK(cudaFreeHost(h_bestgw_pinned));

    SDL_DestroyTexture(heat_tex);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
