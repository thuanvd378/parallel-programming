// ================================================================
// ZIPHUST GPU vs CPU Decompressor Benchmark
// Decompresses .ziphust files using CUDA GPU or OpenMP CPU
// Format: Simple RLE encoding for fair GPU/CPU comparison
// ================================================================

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shlobj.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <atomic>
#include <thread>
#include <sstream>
#include <omp.h>

#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Shell32.lib")
#pragma comment(lib, "Ole32.lib")

// ===========================
// ZIPHUST FORMAT
// ===========================
static constexpr uint32_t ZIPHUST_MAGIC = 0x5448505A; // "ZPHT" little-endian (Z=5A P=50 H=48 T=54)

#pragma pack(push, 1)
struct ZiphustHeader {
    uint32_t magic;           // ZIPHUST_MAGIC
    uint32_t original_size;   // Decompressed size
    uint32_t compressed_size; // Compressed data size
    uint32_t reserved;
};
#pragma pack(pop)

// ===========================
// CUDA CONFIG
// ===========================
#define THREADS_PER_BLOCK 256
#define MAX_FILES_PER_BATCH 10000
#define MAX_FILE_SIZE (256 * 1024)  // 256KB max per file

// ===========================
// CUDA ERROR CHECK
// ===========================
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        return _e; \
    } \
} while (0)

// ===========================
// GPU: RLE Decompression Kernel
// One block per file
// ===========================
__global__ void ziphustDecompressKernel(
    const uint8_t* __restrict__ compressedData,
    uint8_t* __restrict__ decompressedData,
    const uint32_t* __restrict__ compOffsets,
    const uint32_t* __restrict__ compSizes,
    const uint32_t* __restrict__ decompOffsets,
    const uint32_t* __restrict__ decompSizes,
    int numFiles
) {
    int fileId = blockIdx.x;
    if (fileId >= numFiles) return;
    
    // Only thread 0 does the work (RLE is sequential within a file)
    if (threadIdx.x != 0) return;
    
    const uint8_t* src = compressedData + compOffsets[fileId];
    uint8_t* dst = decompressedData + decompOffsets[fileId];
    
    uint32_t srcSize = compSizes[fileId];
    uint32_t dstSize = decompSizes[fileId];
    
    uint32_t srcPos = 0;
    uint32_t dstPos = 0;
    
    while (srcPos < srcSize && dstPos < dstSize) {
        uint8_t token = src[srcPos++];
        
        if (token <= 0x7F) {
            // Literal: copy 'token' bytes
            uint32_t litLen = token;
            for (uint32_t i = 0; i < litLen && srcPos < srcSize && dstPos < dstSize; i++) {
                dst[dstPos++] = src[srcPos++];
            }
        } else if (token <= 0xBF) {
            // Run: repeat next byte (token - 0x80 + 3) times
            uint32_t runLen = (token - 0x80) + 3;
            if (srcPos < srcSize) {
                uint8_t runByte = src[srcPos++];
                for (uint32_t i = 0; i < runLen && dstPos < dstSize; i++) {
                    dst[dstPos++] = runByte;
                }
            }
        }
        // 0xC0-0xFF reserved for future match encoding
    }
}

// ===========================
// CPU: RLE Decompression with OpenMP
// ===========================
static void ziphustDecompressCPU(
    const uint8_t* compressedData,
    uint8_t* decompressedData,
    const uint32_t* compOffsets,
    const uint32_t* compSizes,
    const uint32_t* decompOffsets,
    const uint32_t* decompSizes,
    int numFiles
) {
    #pragma omp parallel for schedule(dynamic)
    for (int fileId = 0; fileId < numFiles; fileId++) {
        const uint8_t* src = compressedData + compOffsets[fileId];
        uint8_t* dst = decompressedData + decompOffsets[fileId];
        
        uint32_t srcSize = compSizes[fileId];
        uint32_t dstSize = decompSizes[fileId];
        
        uint32_t srcPos = 0;
        uint32_t dstPos = 0;
        
        while (srcPos < srcSize && dstPos < dstSize) {
            uint8_t token = src[srcPos++];
            
            if (token <= 0x7F) {
                uint32_t litLen = token;
                for (uint32_t i = 0; i < litLen && srcPos < srcSize && dstPos < dstSize; i++) {
                    dst[dstPos++] = src[srcPos++];
                }
            } else if (token <= 0xBF) {
                uint32_t runLen = (token - 0x80) + 3;
                if (srcPos < srcSize) {
                    uint8_t runByte = src[srcPos++];
                    for (uint32_t i = 0; i < runLen && dstPos < dstSize; i++) {
                        dst[dstPos++] = runByte;
                    }
                }
            }
        }
    }
}

// ===========================
// DPI Awareness
// ===========================
static void EnableDpiAwareness() {
    SetProcessDPIAware();
}

static UINT GetDpi(HWND hwnd) {
    HDC hdc = GetDC(hwnd);
    UINT dpi = GetDeviceCaps(hdc, LOGPIXELSY);
    ReleaseDC(hwnd, hdc);
    return dpi ? dpi : 96;
}

static HFONT CreateUIFont(int pt, UINT dpi) {
    LOGFONTW lf{};
    lf.lfHeight = -MulDiv(pt, (int)dpi, 72);
    lf.lfWeight = FW_NORMAL;
    lf.lfCharSet = DEFAULT_CHARSET;
    lf.lfQuality = CLEARTYPE_QUALITY;
    wcscpy_s(lf.lfFaceName, L"Segoe UI");
    return CreateFontIndirectW(&lf);
}

// ===========================
// File Dialog Helpers
// ===========================
static std::wstring SelectFolderDialogW(HWND owner, const wchar_t* title) {
    wchar_t path[MAX_PATH] = {0};
    BROWSEINFOW bi{};
    bi.hwndOwner = owner;
    bi.lpszTitle = title;
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    
    LPITEMIDLIST pidl = SHBrowseForFolderW(&bi);
    if (!pidl) return L"";
    
    if (SHGetPathFromIDListW(pidl, path)) {
        CoTaskMemFree(pidl);
        return std::wstring(path);
    }
    CoTaskMemFree(pidl);
    return L"";
}

// ===========================
// Helpers
// ===========================
static void AppendText(HWND hEdit, const std::wstring& s) {
    int len = GetWindowTextLengthW(hEdit);
    SendMessageW(hEdit, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessageW(hEdit, EM_REPLACESEL, 0, (LPARAM)s.c_str());
    SendMessageW(hEdit, EM_SCROLLCARET, 0, 0);
}

static std::wstring GetWindowTextWStr(HWND h) {
    int n = GetWindowTextLengthW(h);
    std::wstring s((size_t)n, L'\0');
    GetWindowTextW(h, s.data(), n + 1);
    return s;
}

static std::wstring FormatDouble(double val, int decimals = 2) {
    std::wstringstream ss;
    ss.precision(decimals);
    ss << std::fixed << val;
    return ss.str();
}

// ===========================
// File Info Structure
// ===========================
struct FileInfo {
    std::filesystem::path path;
    uint32_t originalSize;
    uint32_t compressedSize;
    std::vector<uint8_t> compressedData;
};

// ===========================
// GUI Application
// ===========================
enum : UINT {
    IDC_RAD_GPU    = 101,
    IDC_RAD_CPU    = 102,
    IDC_ED_INPUT   = 103,
    IDC_BTN_IN     = 104,
    IDC_ED_OUTDIR  = 105,
    IDC_BTN_OUT    = 106,
    IDC_BTN_START  = 107,
    IDC_PB         = 108,
    IDC_ST_GPU     = 109,
    IDC_ST_STATUS  = 110,
    IDC_ED_LOG     = 111,
};

struct App;
static void ApplyFontAllControls(App* a);

struct App {
    HWND hwnd = nullptr;
    HFONT hFont = nullptr;
    
    HWND radGpu = nullptr;
    HWND radCpu = nullptr;
    HWND edInput = nullptr;
    HWND btnIn = nullptr;
    HWND edOutDir = nullptr;
    HWND btnOut = nullptr;
    HWND btnStart = nullptr;
    HWND pb = nullptr;
    HWND stGpu = nullptr;
    HWND stStatus = nullptr;
    HWND edLog = nullptr;
    
    std::atomic<float> progress{0.0f};
    std::atomic<bool> busy{false};
    std::thread worker;
};

static void ApplyFontAllControls(App* a) {
    HWND ctrls[] = { a->stGpu, a->radGpu, a->radCpu, a->edInput, a->btnIn,
                     a->edOutDir, a->btnOut, a->btnStart, a->stStatus, a->edLog };
    for (HWND h : ctrls) {
        if (h) SendMessageW(h, WM_SETFONT, (WPARAM)a->hFont, TRUE);
    }
}

static void SetEnabledAll(App* a, BOOL en) {
    EnableWindow(a->radGpu, en);
    EnableWindow(a->radCpu, en);
    EnableWindow(a->edInput, en);
    EnableWindow(a->btnIn, en);
    EnableWindow(a->edOutDir, en);
    EnableWindow(a->btnOut, en);
    EnableWindow(a->btnStart, en);
}

// ===========================
// Decompression Job
// ===========================
struct JobResult {
    bool ok = false;
    int fileCount = 0;
    uint64_t totalOriginal = 0;
    uint64_t totalCompressed = 0;
    double elapsedMs = 0;
    std::wstring message;
};

static constexpr UINT WM_APP_LOG = WM_APP + 1;
static constexpr UINT WM_APP_DONE = WM_APP + 2;

static void PostLog(HWND hwnd, const std::wstring& line) {
    auto* p = new std::wstring(line);
    PostMessageW(hwnd, WM_APP_LOG, 0, (LPARAM)p);
}

static cudaError_t DecompressGPU(
    const std::vector<FileInfo>& files,
    const std::filesystem::path& outDir,
    std::atomic<float>& progress,
    JobResult& result
) {
    int numFiles = (int)files.size();
    if (numFiles == 0) return cudaSuccess;
    
    // Calculate total sizes and offsets
    uint64_t totalCompressed = 0;
    uint64_t totalDecompressed = 0;
    
    std::vector<uint32_t> compOffsets(numFiles);
    std::vector<uint32_t> compSizes(numFiles);
    std::vector<uint32_t> decompOffsets(numFiles);
    std::vector<uint32_t> decompSizes(numFiles);
    
    for (int i = 0; i < numFiles; i++) {
        compOffsets[i] = (uint32_t)totalCompressed;
        compSizes[i] = files[i].compressedSize;
        decompOffsets[i] = (uint32_t)totalDecompressed;
        decompSizes[i] = files[i].originalSize;
        totalCompressed += files[i].compressedSize;
        totalDecompressed += files[i].originalSize;
    }
    
    progress.store(10.0f);
    
    // Allocate pinned host memory
    uint8_t* h_compressed = nullptr;
    uint8_t* h_decompressed = nullptr;
    
    CUDA_CHECK(cudaMallocHost(&h_compressed, totalCompressed));
    CUDA_CHECK(cudaMallocHost(&h_decompressed, totalDecompressed));
    
    // Copy compressed data to contiguous buffer
    for (int i = 0; i < numFiles; i++) {
        memcpy(h_compressed + compOffsets[i], files[i].compressedData.data(), files[i].compressedSize);
    }
    
    progress.store(20.0f);
    
    // Allocate device memory
    uint8_t* d_compressed = nullptr;
    uint8_t* d_decompressed = nullptr;
    uint32_t* d_compOffsets = nullptr;
    uint32_t* d_compSizes = nullptr;
    uint32_t* d_decompOffsets = nullptr;
    uint32_t* d_decompSizes = nullptr;
    
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d_compressed, totalCompressed));
    CUDA_CHECK(cudaMalloc(&d_decompressed, totalDecompressed));
    CUDA_CHECK(cudaMalloc(&d_compOffsets, numFiles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_compSizes, numFiles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_decompOffsets, numFiles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_decompSizes, numFiles * sizeof(uint32_t)));
    
    progress.store(30.0f);
    
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpyAsync(d_compressed, h_compressed, totalCompressed, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_compOffsets, compOffsets.data(), numFiles * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_compSizes, compSizes.data(), numFiles * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_decompOffsets, decompOffsets.data(), numFiles * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_decompSizes, decompSizes.data(), numFiles * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    progress.store(40.0f);
    
    // Launch kernel - time this part
    auto t0 = std::chrono::high_resolution_clock::now();
    
    ziphustDecompressKernel<<<numFiles, THREADS_PER_BLOCK, 0, stream>>>(
        d_compressed, d_decompressed,
        d_compOffsets, d_compSizes,
        d_decompOffsets, d_decompSizes,
        numFiles
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsedMs = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    
    progress.store(70.0f);
    
    // Copy back
    CUDA_CHECK(cudaMemcpyAsync(h_decompressed, d_decompressed, totalDecompressed, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    progress.store(85.0f);
    
    // Write output files
    for (int i = 0; i < numFiles; i++) {
        std::filesystem::path outPath = outDir / files[i].path.stem();
        outPath += L".bin";  // Output as .bin
        
        std::ofstream fout(outPath, std::ios::binary);
        if (fout) {
            fout.write((const char*)(h_decompressed + decompOffsets[i]), decompSizes[i]);
        }
    }
    
    progress.store(95.0f);
    
    // Cleanup
    cudaStreamDestroy(stream);
    cudaFreeHost(h_compressed);
    cudaFreeHost(h_decompressed);
    cudaFree(d_compressed);
    cudaFree(d_decompressed);
    cudaFree(d_compOffsets);
    cudaFree(d_compSizes);
    cudaFree(d_decompOffsets);
    cudaFree(d_decompSizes);
    
    result.ok = true;
    result.fileCount = numFiles;
    result.totalOriginal = totalDecompressed;
    result.totalCompressed = totalCompressed;
    
    return cudaSuccess;
}

static void DecompressCPU(
    const std::vector<FileInfo>& files,
    const std::filesystem::path& outDir,
    std::atomic<float>& progress,
    JobResult& result
) {
    int numFiles = (int)files.size();
    if (numFiles == 0) return;
    
    // Calculate total sizes
    uint64_t totalCompressed = 0;
    uint64_t totalDecompressed = 0;
    
    std::vector<uint32_t> compOffsets(numFiles);
    std::vector<uint32_t> compSizes(numFiles);
    std::vector<uint32_t> decompOffsets(numFiles);
    std::vector<uint32_t> decompSizes(numFiles);
    
    for (int i = 0; i < numFiles; i++) {
        compOffsets[i] = (uint32_t)totalCompressed;
        compSizes[i] = files[i].compressedSize;
        decompOffsets[i] = (uint32_t)totalDecompressed;
        decompSizes[i] = files[i].originalSize;
        totalCompressed += files[i].compressedSize;
        totalDecompressed += files[i].originalSize;
    }
    
    progress.store(10.0f);
    
    // Allocate buffers
    std::vector<uint8_t> compressedBuffer(totalCompressed);
    std::vector<uint8_t> decompressedBuffer(totalDecompressed);
    
    for (int i = 0; i < numFiles; i++) {
        memcpy(compressedBuffer.data() + compOffsets[i], files[i].compressedData.data(), files[i].compressedSize);
    }
    
    progress.store(30.0f);
    
    // Decompress with OpenMP - time this part
    auto t0 = std::chrono::high_resolution_clock::now();
    
    ziphustDecompressCPU(
        compressedBuffer.data(),
        decompressedBuffer.data(),
        compOffsets.data(),
        compSizes.data(),
        decompOffsets.data(),
        decompSizes.data(),
        numFiles
    );
    
    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsedMs = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    
    progress.store(70.0f);
    
    // Write output files
    #pragma omp parallel for
    for (int i = 0; i < numFiles; i++) {
        std::filesystem::path outPath = outDir / files[i].path.stem();
        outPath += L".bin";
        
        std::ofstream fout(outPath, std::ios::binary);
        if (fout) {
            fout.write((const char*)(decompressedBuffer.data() + decompOffsets[i]), decompSizes[i]);
        }
    }
    
    progress.store(95.0f);
    
    result.ok = true;
    result.fileCount = numFiles;
    result.totalOriginal = totalDecompressed;
    result.totalCompressed = totalCompressed;
}

static void StartJob(App* a) {
    if (a->busy.load()) return;
    
    const bool useGpu = (SendMessageW(a->radGpu, BM_GETCHECK, 0, 0) == BST_CHECKED);
    
    std::wstring inDirW = GetWindowTextWStr(a->edInput);
    std::wstring outDirW = GetWindowTextWStr(a->edOutDir);
    
    if (inDirW.empty() || outDirW.empty()) {
        MessageBoxW(a->hwnd, L"Please select input and output folders.", L"ZIPHUST", MB_ICONWARNING);
        return;
    }
    
    std::filesystem::path inDir(inDirW);
    std::filesystem::path outDir(outDirW);
    
    if (!std::filesystem::exists(inDir) || !std::filesystem::is_directory(inDir)) {
        MessageBoxW(a->hwnd, L"Invalid input folder.", L"ZIPHUST", MB_ICONWARNING);
        return;
    }
    
    std::filesystem::create_directories(outDir);
    
    a->busy.store(true);
    a->progress.store(0.0f);
    SetEnabledAll(a, FALSE);
    SendMessageW(a->pb, PBM_SETPOS, 0, 0);
    
    AppendText(a->edLog, L"------------------------------------------------------------\r\n");
    AppendText(a->edLog, useGpu ? L"Mode: GPU CUDA\r\n" : L"Mode: CPU OpenMP\r\n");
    AppendText(a->edLog, (L"Input: " + inDirW + L"\r\n"));
    AppendText(a->edLog, (L"Output: " + outDirW + L"\r\n"));
    AppendText(a->edLog, L"Loading files...\r\n");
    
    if (a->worker.joinable()) a->worker.join();
    
    a->worker = std::thread([a, useGpu, inDir, outDir]() {
        auto* result = new JobResult();
        
        // Load all .ziphust files
        std::vector<FileInfo> files;
        
        int totalScanned = 0;
        for (const auto& entry : std::filesystem::directory_iterator(inDir)) {
            totalScanned++;
            
            if (!entry.is_regular_file()) continue;
            
            auto filename = entry.path().filename().wstring();
            auto ext = entry.path().extension().wstring();
            
            // Debug log first 5 files
            if (totalScanned <= 5) {
                PostLog(a->hwnd, L"Scanning: " + filename + L" (ext: " + ext + L")\r\n");
            }
            
            // Convert to lowercase for case-insensitive comparison
            std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
            
            if (ext == L".ziphust") {
                FileInfo fi;
                fi.path = entry.path();
                
                std::ifstream fin(entry.path(), std::ios::binary);
                if (!fin) {
                    PostLog(a->hwnd, L"Failed to open: " + filename + L"\r\n");
                    continue;
                }
                
                ZiphustHeader hdr;
                fin.read((char*)&hdr, sizeof(hdr));
                
                if (hdr.magic != ZIPHUST_MAGIC) {
                    // Debug: show actual vs expected
                    wchar_t buf[100];
                    swprintf_s(buf, L"Invalid magic in %s: got 0x%08X, expected 0x%08X\r\n", 
                              filename.c_str(), hdr.magic, ZIPHUST_MAGIC);
                    PostLog(a->hwnd, buf);
                    continue;
                }
                
                fi.originalSize = hdr.original_size;
                fi.compressedSize = hdr.compressed_size;
                fi.compressedData.resize(hdr.compressed_size);
                fin.read((char*)fi.compressedData.data(), hdr.compressed_size);
                
                files.push_back(std::move(fi));
            }
        }
        
        PostLog(a->hwnd, L"Scanned " + std::to_wstring(totalScanned) + L" items\r\n");
        
        PostLog(a->hwnd, L"Found " + std::to_wstring(files.size()) + L" files\r\n");
        
        if (files.empty()) {
            result->message = L"No .ziphust files found in folder.\r\n";
            PostMessageW(a->hwnd, WM_APP_DONE, (WPARAM)result, 0);
            return;
        }
        
        PostLog(a->hwnd, L"Decompressing...\r\n");
        
        if (useGpu) {
            cudaError_t err = DecompressGPU(files, outDir, a->progress, *result);
            if (err != cudaSuccess) {
                result->ok = false;
                std::string errStr = cudaGetErrorString(err);
                result->message = L"CUDA Error: " + std::wstring(errStr.begin(), errStr.end()) + L"\r\n";
            }
        } else {
            DecompressCPU(files, outDir, a->progress, *result);
        }
        
        PostMessageW(a->hwnd, WM_APP_DONE, (WPARAM)result, 0);
    });
}

// ===========================
// Layout
// ===========================
static void Layout(App* a, int W, int H) {
    const int pad = 12;
    const int rowH = 26;
    const int btnW = 120;
    const int editH = 24;
    
    int x = pad;
    int y = pad;
    
    MoveWindow(a->stGpu, x, y, W - 2 * pad, rowH, TRUE);
    y += rowH + 8;
    
    MoveWindow(a->radGpu, x, y, 140, rowH, TRUE);
    MoveWindow(a->radCpu, x + 150, y, 160, rowH, TRUE);
    y += rowH + 10;
    
    MoveWindow(a->edInput, x, y, W - 3 * pad - btnW, editH, TRUE);
    MoveWindow(a->btnIn, x + (W - 3 * pad - btnW) + pad, y, btnW, editH, TRUE);
    y += editH + 8;
    
    MoveWindow(a->edOutDir, x, y, W - 3 * pad - btnW, editH, TRUE);
    MoveWindow(a->btnOut, x + (W - 3 * pad - btnW) + pad, y, btnW, editH, TRUE);
    y += editH + 10;
    
    MoveWindow(a->btnStart, x, y, 140, 30, TRUE);
    MoveWindow(a->stStatus, x + 150, y + 4, W - x - 150 - pad, rowH, TRUE);
    y += 40;
    
    MoveWindow(a->pb, x, y, W - 2 * pad, 18, TRUE);
    y += 18 + 10;
    
    MoveWindow(a->edLog, x, y, W - 2 * pad, H - y - pad, TRUE);
}

// ===========================
// Window Procedure
// ===========================
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    App* a = (App*)GetWindowLongPtrW(hwnd, GWLP_USERDATA);
    
    switch (msg) {
    case WM_CREATE: {
        auto* cs = (CREATESTRUCTW*)lParam;
        a = (App*)cs->lpCreateParams;
        a->hwnd = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)a);
        
        a->hFont = CreateUIFont(10, GetDpi(hwnd));
        
        // Get GPU name
        std::wstring gpuName = L"(no CUDA device)";
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            gpuName = std::wstring(prop.name, prop.name + strlen(prop.name));
        }
        
        a->stGpu = CreateWindowW(L"STATIC", (L"GPU: " + gpuName).c_str(), WS_CHILD | WS_VISIBLE,
                                 0,0,0,0, hwnd, (HMENU)IDC_ST_GPU, nullptr, nullptr);
        
        a->radGpu = CreateWindowW(L"BUTTON", L"GPU CUDA", WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
                                  0,0,0,0, hwnd, (HMENU)IDC_RAD_GPU, nullptr, nullptr);
        a->radCpu = CreateWindowW(L"BUTTON", L"CPU OpenMP", WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
                                  0,0,0,0, hwnd, (HMENU)IDC_RAD_CPU, nullptr, nullptr);
        
        SendMessageW(a->radGpu, BM_SETCHECK, BST_CHECKED, 0);
        
        a->edInput = CreateWindowW(L"EDIT", L"", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                                   0,0,0,0, hwnd, (HMENU)IDC_ED_INPUT, nullptr, nullptr);
        a->btnIn = CreateWindowW(L"BUTTON", L"Input Folder...", WS_CHILD | WS_VISIBLE,
                                 0,0,0,0, hwnd, (HMENU)IDC_BTN_IN, nullptr, nullptr);
        
        a->edOutDir = CreateWindowW(L"EDIT", L"", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                                    0,0,0,0, hwnd, (HMENU)IDC_ED_OUTDIR, nullptr, nullptr);
        a->btnOut = CreateWindowW(L"BUTTON", L"Output Folder...", WS_CHILD | WS_VISIBLE,
                                  0,0,0,0, hwnd, (HMENU)IDC_BTN_OUT, nullptr, nullptr);
        
        a->btnStart = CreateWindowW(L"BUTTON", L"Decompress", WS_CHILD | WS_VISIBLE,
                                    0,0,0,0, hwnd, (HMENU)IDC_BTN_START, nullptr, nullptr);
        
        a->stStatus = CreateWindowW(L"STATIC", L"Ready.", WS_CHILD | WS_VISIBLE,
                                    0,0,0,0, hwnd, (HMENU)IDC_ST_STATUS, nullptr, nullptr);
        
        a->pb = CreateWindowW(PROGRESS_CLASSW, L"", WS_CHILD | WS_VISIBLE,
                              0,0,0,0, hwnd, (HMENU)IDC_PB, nullptr, nullptr);
        SendMessageW(a->pb, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        
        a->edLog = CreateWindowW(L"EDIT", L"", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | WS_VSCROLL | ES_READONLY,
                                 0,0,0,0, hwnd, (HMENU)IDC_ED_LOG, nullptr, nullptr);
        
        ApplyFontAllControls(a);
        
        // Timer for progress updates
        SetTimer(hwnd, 1, 80, nullptr);
        
        AppendText(a->edLog, L"ZIPHUST GPU vs CPU Decompressor\r\n");
        AppendText(a->edLog, L"Select input folder with .ziphust files\r\n");
        AppendText(a->edLog, L"------------------------------------------------------------\r\n");
        
        int numThreads = omp_get_max_threads();
        AppendText(a->edLog, L"CPU threads: " + std::to_wstring(numThreads) + L"\r\n");
        
        return 0;
    }
    case WM_SIZE: {
        if (!a) break;
        Layout(a, LOWORD(lParam), HIWORD(lParam));
        return 0;
    }
    case WM_COMMAND: {
        if (!a) break;
        int id = LOWORD(wParam);
        
        if (id == IDC_BTN_IN) {
            std::wstring p = SelectFolderDialogW(hwnd, L"Select input folder with .ziphust files");
            if (!p.empty()) SetWindowTextW(a->edInput, p.c_str());
        } else if (id == IDC_BTN_OUT) {
            std::wstring p = SelectFolderDialogW(hwnd, L"Select output folder");
            if (!p.empty()) SetWindowTextW(a->edOutDir, p.c_str());
        } else if (id == IDC_BTN_START) {
            StartJob(a);
        }
        return 0;
    }
    case WM_TIMER: {
        if (!a) break;
        float p = a->progress.load();
        SendMessageW(a->pb, PBM_SETPOS, (WPARAM)(int)p, 0);
        return 0;
    }
    case WM_APP_LOG: {
        if (!a) break;
        auto* ps = (std::wstring*)lParam;
        if (ps) {
            AppendText(a->edLog, *ps);
            delete ps;
        }
        return 0;
    }
    case WM_APP_DONE: {
        if (!a) break;
        auto* result = (JobResult*)wParam;
        
        a->busy.store(false);
        a->progress.store(100.0f);
        SendMessageW(a->pb, PBM_SETPOS, 100, 0);
        
        if (result) {
            if (!result->ok) {
                SetWindowTextW(a->stStatus, L"Error.");
                AppendText(a->edLog, L"[FAILED]\r\n");
                if (!result->message.empty()) AppendText(a->edLog, result->message);
            } else {
                SetWindowTextW(a->stStatus, L"Completed.");
                AppendText(a->edLog, L"[SUCCESS]\r\n");
                AppendText(a->edLog, L"Files: " + std::to_wstring(result->fileCount) + L"\r\n");
                AppendText(a->edLog, L"Decompression time: " + FormatDouble(result->elapsedMs, 2) + L" ms\r\n");
                
                double mbps = (double)result->totalOriginal / 1024.0 / 1024.0 / (result->elapsedMs / 1000.0);
                double filesPerSec = (double)result->fileCount / (result->elapsedMs / 1000.0);
                
                AppendText(a->edLog, L"Speed: " + FormatDouble(mbps, 1) + L" MB/s\r\n");
                AppendText(a->edLog, L"Throughput: " + FormatDouble(filesPerSec, 0) + L" files/sec\r\n");
            }
            delete result;
        }
        
        SetEnabledAll(a, TRUE);
        return 0;
    }
    case WM_DESTROY: {
        if (a) {
            if (a->worker.joinable()) a->worker.join();
            if (a->hFont) DeleteObject(a->hFont);
        }
        PostQuitMessage(0);
        return 0;
    }
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// ===========================
// Entry Point
// ===========================
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int nCmdShow) {
    EnableDpiAwareness();
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    
    INITCOMMONCONTROLSEX icc{};
    icc.dwSize = sizeof(icc);
    icc.dwICC = ICC_PROGRESS_CLASS;
    InitCommonControlsEx(&icc);
    
    App app{};
    
    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = L"ZIPHUST_GPU_CPU";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClassW(&wc);
    
    HWND hwnd = CreateWindowW(
        wc.lpszClassName,
        L"ZIPHUST Decompressor - GPU vs CPU Benchmark",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        900, 650,
        nullptr, nullptr, hInst, &app
    );
    
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);
    
    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    
    CoUninitialize();
    return (int)msg.wParam;
}
