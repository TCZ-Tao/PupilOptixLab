#pragma once
#include <fstream>
#include <filesystem>

#include <vector>
#include <unordered_map>

#include "util/log.h"
#include "util/aabb.h"

#include "cuda.h"
#include "cuda/data_view.h"

// number of elements
#define PLY_3DGS_NUM_POS 3
#define PLY_3DGS_NUM_N 3
#define PLY_3DGS_NUM_SHS 48
#define PLY_3DGS_NUM_OPA 1
#define PLY_3DGS_NUM_SCALE 3
#define PLY_3DGS_NUM_ROT 4

#define PLY_3DGS_ALPHA_MIN 0.004f // minimum response for bounding

#define PLY_3DGS_CHUNK_SIZE 16

namespace Pupil::resource {

class PlyLoader : public util::Singleton<PlyLoader> {
public:
    struct PlyProperty {
        std::string type;
        std::string name;
    };
    
    struct PlyHeader {
        std::string format;
        unsigned int numVertices;
        unsigned int numFaces;
        std::vector<PlyProperty> vertexProperties;
    };
    
    // per vertex data
    struct PlyVertexStorage {
        float position[3];
        float normal[3];
        float shs[48];
        float opacity;
        float scale[3];
        float rotation[4];
    };

    struct ThreedgsDeviceMemory {
        CUdeviceptr position = 0;
        CUdeviceptr opacity = 0;
        CUdeviceptr scale = 0;
        CUdeviceptr rotation = 0;
        CUdeviceptr shs = 0;
    };

    // 3dgs points buffer
    struct PointData3dgs {
        std::vector<float> position_vec;
        //std::vector<float> normal_vec;
        std::vector<float> shs_vec;
        std::vector<float> opacity_vec;
        std::vector<float> scale_vec;
        std::vector<float> rotation_vec;

        ThreedgsDeviceMemory device_memory;

        ~PointData3dgs() noexcept;
    };

    void LoadFromFile(std::string_view) noexcept;

    PlyHeader header;

    std::unordered_map<std::string, std::unique_ptr<PointData3dgs>,
                       util::StringHash, std::equal_to<>>
        m_3dgs_data; // todo:记得处理销毁（没看到他在哪调用shapemanager.Clear，没法做）

    // bounding buffer
    std::vector<float> boundMeshPos_vec;
    std::vector<uint32_t> boundMeshIdx_vec;
    util::AABB aabb;

private:
    void LoadPlyHeader(std::ifstream& plyFile);
    void LoadPoints(std::ifstream& plyFile, std::string_view file_path);
    void CreateBounding(std::string_view file_path);
};
}// namespace Pupil::resource
