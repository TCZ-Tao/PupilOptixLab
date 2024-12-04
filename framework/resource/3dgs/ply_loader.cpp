#include "ply_loader.h"

#include "cuda/util.h"

#include "util/type.h" // vector, matrix
#include "util/transform.h"

#include "world/world.h"
#include "system/buffer.h"
#include "system/gui/gui.h"
#include "system/system.h"


namespace Pupil::resource {
void PlyLoader::LoadFromFile(std::string_view file_path) noexcept{
    std::ifstream plyFile(file_path.data(), std::ios::binary);

    LoadPlyHeader(plyFile);
    LoadPoints(plyFile, file_path);

    CreateBounding(file_path);
}

void PlyLoader::LoadPlyHeader(std::ifstream& plyFile) {
    std::string line;
    bool        headerEnd = false;

    while (std::getline(plyFile, line)) {
        std::istringstream iss(line);
        std::string        token;
        iss >> token;
        if (token == "ply") {
        } else if (token == "format") {
            iss >> header.format;
        } else if (token == "element") {
            iss >> token;
            if (token == "vertex") iss >> header.numVertices;
            if (token == "face") {
                iss >> header.numFaces;
                if (header.numFaces > 0) 
                    Pupil::Log::Error("3DGS PLY file should not have faces");
            }
        } else if (token == "property") {
            PlyProperty property;
            iss >> property.type >> property.name;
            if (header.vertexProperties.size() < static_cast<size_t>(header.numVertices))
                header.vertexProperties.push_back(property);
            else
                Pupil::Log::Error("3DGS PLY file should not have faces");
        } else if (token == "end_header") {
            headerEnd = true;
            break;
        }
    }
    if (!headerEnd) Pupil::Log::Error("Cound not find end_header");
}

void PlyLoader::LoadPoints(std::ifstream& plyFile, std::string_view file_path) {
    // allocate buffer space
    auto point_data = std::make_unique<PointData3dgs>();

    point_data->position_vec.resize(sizeof(float) * PLY_3DGS_NUM_POS * header.numVertices);
    //buffer.normal_vec.resize(sizeof(float) * PLY_3DGS_NUM_N * header.numVertices);
    point_data->shs_vec.resize(sizeof(float) * PLY_3DGS_NUM_SHS * header.numVertices);
    point_data->opacity_vec.resize(sizeof(float) * PLY_3DGS_NUM_OPA * header.numVertices);
    point_data->scale_vec.resize(sizeof(float) * PLY_3DGS_NUM_SCALE * header.numVertices);
    point_data->rotation_vec.resize(sizeof(float) * PLY_3DGS_NUM_ROT * header.numVertices);

    for (auto i = 0u; i < header.numVertices; ++i) {
        static_assert(sizeof(PlyVertexStorage) == 62 * sizeof(float));
        assert(!plyFile.eof());

        PlyVertexStorage vertexStorage;
        plyFile.read(reinterpret_cast<char*>(&vertexStorage), sizeof(vertexStorage));
        
        assert(vertexStorage.normal[0] == 0.f);
        assert(vertexStorage.normal[1] == 0.f);
        assert(vertexStorage.normal[2] == 0.f);

        for (int j = 0; j < PLY_3DGS_NUM_POS; ++j)
            point_data->position_vec[i * PLY_3DGS_NUM_POS + j] = vertexStorage.position[j];
        //for (int j = 0; j < PLY_3DGS_NUM_N; ++j)
        //    point_data->normal_vec[i * PLY_3DGS_NUM_N + j] = vertexStorage.normal[j];
        for (int j = 0; j < PLY_3DGS_NUM_SHS; ++j)
            point_data->shs_vec[i * PLY_3DGS_NUM_SHS + j] = vertexStorage.shs[j]; //不改排序了
        point_data->opacity_vec[i * PLY_3DGS_NUM_OPA] = 1.f / (1.f + std::exp(-vertexStorage.opacity));
        for (int j = 0; j < PLY_3DGS_NUM_SCALE; ++j)
            point_data->scale_vec[i * PLY_3DGS_NUM_SCALE + j] = std::exp(vertexStorage.scale[j]);

        float invSqrt = 1.f / std::sqrt(vertexStorage.rotation[0] * vertexStorage.rotation[0] 
            + vertexStorage.rotation[1] * vertexStorage.rotation[1] 
            + vertexStorage.rotation[2] * vertexStorage.rotation[2] 
            + vertexStorage.rotation[3] * vertexStorage.rotation[3]);
        for (int j = 0; j < PLY_3DGS_NUM_ROT; ++j)
            point_data->rotation_vec[i * PLY_3DGS_NUM_ROT + j] = vertexStorage.rotation[j] * invSqrt;
    }

    // upload to device
    point_data->device_memory.position = cuda::CudaMemcpyToDevice(
        point_data->position_vec.data(), point_data->position_vec.size() * sizeof(float));
    point_data->device_memory.opacity = cuda::CudaMemcpyToDevice(
        point_data->opacity_vec.data(), point_data->opacity_vec.size() * sizeof(float));
    point_data->device_memory.scale = cuda::CudaMemcpyToDevice(
        point_data->scale_vec.data(), point_data->scale_vec.size() * sizeof(float));
    point_data->device_memory.rotation = cuda::CudaMemcpyToDevice(
        point_data->rotation_vec.data(), point_data->rotation_vec.size() * sizeof(float));
    point_data->device_memory.shs = cuda::CudaMemcpyToDevice(
        point_data->shs_vec.data(), point_data->shs_vec.size() * sizeof(float));

    m_3dgs_data.emplace(file_path, std::move(point_data));
    Pupil::Log::Info("Loaded 3DGS PLY file with {} points", header.numVertices);
}


void PlyLoader::CreateBounding(std::string_view file_path) {
    boundMeshPos_vec.reserve(header.numVertices * 12);
    boundMeshIdx_vec.reserve(header.numVertices * 60);

    auto point_data = m_3dgs_data.find(file_path);

    for (auto i = 0u; i < header.numVertices; ++i) {
    //for (auto i = 0u; i < 100; ++i) {
        // create a unit icosahedron
        // https://github.com/alexisgea/sphere_generator/blob/
        // master/Assets/SphereGenerator/Scripts/Platonics/Icosahedron.cs
        float phi = (1.f + sqrt(5.f)) / 2.f;
        // 12
        std::vector<float> icosaPos{
            -1.f, phi, 0.f, 1.f, phi, 0.f, -1.f, -phi, 0.f, 1.f, -phi, 0.f,
            0.f, -1.f, phi, 0.f, 1.f, phi, 0.f, -1.f, -phi, 0.f, 1.f, -phi,
            phi, 0.f, -1.f, phi, 0.f, 1.f, -phi, 0.f, -1.f, -phi, 0.f, 1.f
        };

        // 20*3=60
        std::vector<uint32_t> icosaIdx{
            0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
            1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
            3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
            4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1
        };

        // transform into stretched
        std::vector<float> icosaPos_stretched;
        icosaPos_stretched.reserve(12);
        for (int j = 0; j < icosaPos.size() / 3; ++j) {
            Pupil::util::Float3 v(icosaPos[j*3], icosaPos[j*3 + 1], icosaPos[j*3 + 2]);
            v = v * sqrt(2.f * log(point_data->second->opacity_vec[i * PLY_3DGS_NUM_OPA] / PLY_3DGS_ALPHA_MIN));
            // from quaternion to matrix
            Pupil::util::Float4 q(point_data->second->rotation_vec[i * PLY_3DGS_NUM_ROT],
                                  point_data->second->rotation_vec[i * PLY_3DGS_NUM_ROT + 1],
                                  point_data->second->rotation_vec[i * PLY_3DGS_NUM_ROT + 2],
                                  point_data->second->rotation_vec[i * PLY_3DGS_NUM_ROT + 3]);
            Pupil::util::Mat4 rotate;
            float r = q.x;
            float x = q.y;
            float y = q.z;
            float z = q.w;
            rotate.re[0][0] = 1.f - 2.f * (y * y - z * z);
            rotate.re[0][1] = 2.f * (x * y - z * r);
            rotate.re[0][2] = 2.f * (x * z + y * r);
            rotate.re[1][0] = 2.f * (x * y - z * r);
            rotate.re[1][1] = 1.f - 2.f * (x * x + z * z);
            rotate.re[1][2] = 2.f * (y * z - x * r);
            rotate.re[2][0] = 2.f * (x * z - y * r);
            rotate.re[2][1] = 2.f * (y * z + x * r);
            rotate.re[2][2] = 1.f - 2.f * (x * x + y * y);
            rotate.re[3][3] = 1.f;

            rotate.r1 *= -1.f;
            rotate.r2 *= -1.f;

            Pupil::util::Mat4 scale(
                point_data->second->scale_vec[i * PLY_3DGS_NUM_SCALE], 0.f, 0.f, 0.f, 
                0.f, point_data->second->scale_vec[i * PLY_3DGS_NUM_SCALE + 1], 0.f, 0.f, 
                0.f, 0.f, point_data->second->scale_vec[i * PLY_3DGS_NUM_SCALE + 2], 0.f, 
                0.f, 0.f, 0.f, 1.f);

            //// RSv
            Pupil::util::Mat4 RS(rotate * scale);
            v = Pupil::util::Transform::TransformVector(v, RS);

            // translate to center of the particle
            v += Pupil::util::Float3(point_data->second->position_vec[i * PLY_3DGS_NUM_POS],
                                     point_data->second->position_vec[i * PLY_3DGS_NUM_POS + 1],
                                     point_data->second->position_vec[i * PLY_3DGS_NUM_POS + 2]);

            icosaPos_stretched.push_back(v.x);
            icosaPos_stretched.push_back(v.y);
            icosaPos_stretched.push_back(v.z);

            aabb.Merge(v);
        }
        // add vert and idx to the whole list
        int size_current = boundMeshPos_vec.size() / 3;
        for (int j = 0; j < icosaPos_stretched.size(); ++j)
            boundMeshPos_vec.push_back(icosaPos_stretched[j]);
        for (int j = 0; j < icosaIdx.size(); ++j)
            boundMeshIdx_vec.push_back(icosaIdx[j] + size_current);
    }

    Pupil::Log::Info("Bounding Created with {} vertices, {} indices", 
        boundMeshPos_vec.size() / 3, boundMeshIdx_vec.size());

    //for (int i = 0; i < boundMeshIdx_vec.size(); ++i) {
    //    Pupil::Log::Info("indices {}", boundMeshIdx_vec[i]);
    //}
}


PlyLoader::PointData3dgs::~PointData3dgs() noexcept {
    CUDA_FREE(device_memory.position);
    CUDA_FREE(device_memory.opacity);
    CUDA_FREE(device_memory.scale);
    CUDA_FREE(device_memory.rotation);
    CUDA_FREE(device_memory.shs);
}

}// namespace Pupil::resource
