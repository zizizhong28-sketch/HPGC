#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <map>
#include <queue>
#include <cmath>
#include <algorithm>
#include <random>
#include <tuple>
#include <stdexcept>
#include <memory>
#include <sys/time.h>

using namespace std;
namespace py = pybind11;

// Forward declarations
struct Octant;
struct Node;
struct process_info;

struct Octant {
    vector<shared_ptr<Octant>> children;
    vector<float> center;
    float extent;
    float z_extent;
    int depth;
    int curind;
    bool is_leaf;
    int octant;
    int parent;
    int nodeid;
    vector<int> point_indices; // 新增：记录属于当前节点的点云索引

    Octant(
        vector<shared_ptr<Octant>> _children,
        vector<float> _center,
        float _extent,
        float _z_extent,
        int _depth,
        bool _is_leaf,
        int _parent,
        int _nodeid,
        vector<int> _point_indices = {}
    ) : children(_children), center(_center), extent(_extent), z_extent(_z_extent),depth(_depth), is_leaf(_is_leaf), parent(_parent),nodeid(_nodeid),point_indices(_point_indices) {
        octant = 0;
        curind = 0;
    }

    // 深拷贝构造函数
    Octant(const Octant& other) {
        // 深拷贝 center
        this->center = other.center;

        // 深拷贝 children
        this->children.reserve(other.children.size());
        for (const auto& child : other.children) {
            if (child) {
                this->children.push_back(std::make_shared<Octant>(*child));
            } else {
                this->children.push_back(nullptr);
            }
        }

        // 拷贝其他成员
        this->extent = other.extent;
        this->z_extent = other.z_extent;
        this->depth = other.depth;
        this->is_leaf = other.is_leaf;
        this->octant = other.octant;
        this->curind = other.curind;
        this->parent = other.parent;
        this->nodeid = other.nodeid;
        this->point_indices = other.point_indices;
    }

    // 赋值操作符
    Octant& operator=(const Octant& other) {
        if (this == &other) return *this;

        // 深拷贝 center
        this->center = other.center;

        // 深拷贝 children
        this->children.clear();
        this->children.reserve(other.children.size());
        for (const auto& child : other.children) {
            if (child) {
                this->children.push_back(std::make_shared<Octant>(*child));
            } else {
                this->children.push_back(nullptr);
            }
        }

        // 拷贝其他成员
        this->extent = other.extent;
        this->z_extent = other.z_extent;
        this->depth = other.depth;
        this->is_leaf = other.is_leaf;
        this->octant = other.octant;
        this->curind = other.curind;
        this->parent = other.parent;
        this->nodeid = other.nodeid;
        this->point_indices = other.point_indices;

        return *this;
    }
};

// Node structure for processing octree
struct Node {
    int index;
    int fatherIndex;
    map<int, vector<int>> dis_n_neibours;
    vector<float> location;
    int depth;
    int curIndex;
    int parentOccupancy;
    int occupancy;

    Node() {
        index = -1;
        fatherIndex = -1;
        dis_n_neibours.clear();
        location = { 0.1f, 0.1f, 0.1f };
        depth = -1;
        curIndex = 0;
        parentOccupancy = 0;
        occupancy = 0;
    }
};

// Process info structure
struct process_info {
    map<int, Node> nodedict;
    map<int, int> layerIndexs;
    int maxIndex;
    int maxLayer;

    process_info() {
        nodedict.clear();
        layerIndexs.clear();
        maxIndex = -1;
        maxLayer = 0;
    }
};


class COctree {
    public:
        std::vector<std::shared_ptr<Octant>> node;
        int level = 0;
    
        COctree() = default;
        explicit COctree(int lvl) : level(lvl) {}
    };


// Convert octree to point cloud
vector<vector<float>> octree2pointcloud(shared_ptr<Octant> root) {
    vector<vector<float>> points;

    if (!root) {
        return points;
    }

    // Recursive helper to collect points
    function<void(shared_ptr<Octant>)> leaf_DFS = [&](shared_ptr<Octant> node) {
        if (!node) return;
        if (node->is_leaf) {
            points.push_back(node->center);
            return;
        }
        for (auto& child : node->children) {
            if (child) {
                leaf_DFS(child);
            }
        }
    };

    leaf_DFS(root);
    return points;
}



std::tuple<std::vector<std::shared_ptr<COctree>>, std::shared_ptr<Octant>, std::vector<int>>
GenOctree(const std::vector<std::vector<float>>& db_np, int max_layer, const std::vector<float> new_center = std::vector<float>(3, 0.5f), float new_extent = 0.5f, float z_rate = 1, bool cylin = true) {
    if (db_np.empty()) {
        return {std::vector<std::shared_ptr<COctree>>{}, nullptr, std::vector<int>{}};
    }
    int N = db_np.size();
    int dim = db_np[0].size();
    std::vector<float> db_np_min(dim, INFINITY);
    std::vector<float> db_np_max(dim, -INFINITY);

    for (const auto& point : db_np) {
        for (int i = 0; i < dim; ++i) {
            db_np_min[i] = std::min(db_np_min[i], point[i]);
            db_np_max[i] = std::max(db_np_max[i], point[i]);
        }
    }
    float db_extent = new_extent;
    std::vector<float> db_center = new_center;
    float max_r;
    float db_z_extent;
    if (cylin) {
        db_z_extent = z_rate * db_extent;
        max_r = std::max(db_np_max[0], db_np_max[1]);
    } else {
        db_z_extent = new_extent;
        max_r = new_extent + db_center[0];
    }
    std::vector<int> point_indices(N);
    std::iota(point_indices.begin(), point_indices.end(), 0);
    int nodeid = 0;
    auto root = std::make_shared<Octant>(
        std::vector<std::shared_ptr<Octant>>(8),
        db_center,
        db_extent,
        db_z_extent,
        1,
        true,
        nodeid,
        nodeid,
        point_indices
    );
    std::queue<std::shared_ptr<Octant>> octantQueue;
    octantQueue.push(root);
    std::vector<int> total_nodeNumlist;
    std::vector<std::shared_ptr<COctree>> Octree(max_layer + 1);
    for (int i = 0; i <= max_layer; ++i) {
        Octree[i] = std::make_shared<COctree>(i);
    }
    Octree[0]->node.push_back(root);
    int cur_layer = 1;

    // 3D Hilbert curve lookup table for transforming Morton code to Hilbert order
    // This table defines the Hilbert order for each possible 3-bit Morton code (0-7)
    const std::vector<int> mortonToHilbert = {
        0, 1, 3, 2, 7, 6, 4, 5
    };

    while (!octantQueue.empty()) {
        int cur_len = octantQueue.size();
        total_nodeNumlist.push_back(cur_len);
        Octree[cur_layer]->level = cur_layer;
        for (int i = 0; i < cur_len; ++i) {
            auto node = octantQueue.front();
            octantQueue.pop();
            if (node->depth >= max_layer) {
                continue;
            }
            float effective_radius = node->center[0];
            if ((effective_radius <= max_r / 4 && node->depth >= max_layer - 2) ||
                (effective_radius > max_r / 4 && effective_radius <= max_r / 2 && node->depth >= max_layer - 1)) {
                continue;
            }
            std::vector<std::vector<int>> children_point_indices(8);
            int occupancy = 0;
            for (int point_idx : node->point_indices) {
                const auto& point = db_np[point_idx];
                int morton_code = 0;
                if (point[0] > node->center[0]) morton_code |= 1;
                if (point[1] > node->center[1]) morton_code |= 2;
                if (point[2] > node->center[2]) morton_code |= 4;
                
                // Convert Morton code to Hilbert order
                int hilbert_index = mortonToHilbert[morton_code];
                children_point_indices[hilbert_index].push_back(point_idx);
            }
            int child_num = 0;
            for (int i = 0; i < 8; ++i) {
                if (!children_point_indices[i].empty()) {
                    occupancy += (1 << (7 - i));
                    child_num++;
                }
            }
            node->octant = occupancy;
            node->is_leaf = false;
            std::vector<float> factor = {-0.5f, 0.5f};
            for (int i = 0; i < 8; ++i) {
                if (!children_point_indices[i].empty()) {
                    // Map Hilbert index back to original Morton code for child center calculation
                    int morton_code = std::distance(mortonToHilbert.begin(), std::find(mortonToHilbert.begin(), mortonToHilbert.end(), i));
                    std::vector<float> child_center(3);
                    child_center[0] = node->center[0] + factor[(morton_code & 1) ? 1 : 0] * node->extent;
                    child_center[1] = node->center[1] + factor[(morton_code & 2) ? 1 : 0] * node->extent;
                    child_center[2] = node->center[2] + factor[(morton_code & 4) ? 1 : 0] * node->z_extent;
                    float child_extent = 0.5f * node->extent;
                    float child_z_extent = 0.5f * node->z_extent;
                    nodeid++;
                    auto child = std::make_shared<Octant>(
                        std::vector<std::shared_ptr<Octant>>(8),
                        child_center,
                        child_extent,
                        child_z_extent,
                        node->depth + 1,
                        true,
                        node->nodeid,
                        nodeid,
                        children_point_indices[i]
                    );
                    child->curind = i;
                    node->children[i] = child;
                    octantQueue.push(child);
                    Octree[cur_layer]->node.push_back(child);
                }
            }
        }
        cur_layer++;
    }
    Octree.pop_back();
    return {Octree, root, total_nodeNumlist};
}



// 计算 Octant 节点数量的函数
int NodeNumCount(const std::shared_ptr<Octant>& root) {
    if (!root) return 0;
    std::queue<std::shared_ptr<Octant>> node_queue;
    node_queue.push(root);
    int node_num = 1;
    while (!node_queue.empty()) {
        auto top = node_queue.front();
        node_queue.pop();
        for (const auto& child : top->children) {
            if (child) {
                node_queue.push(child);
                node_num++;
            }
        }
    }
    return node_num;
}

// 计算每一层 Octant 节点数量的函数
std::vector<int> NodeNumCountPerLevel(const std::shared_ptr<Octant>& root) {
    std::vector<int> levelCounts;
    if (!root) return levelCounts;
    std::queue<std::shared_ptr<Octant>> nodeQueue;
    nodeQueue.push(root);
    while (!nodeQueue.empty()) {
        int levelSize = nodeQueue.size();
        levelCounts.push_back(levelSize);
        for (int i = 0; i < levelSize; ++i) {
            auto current = nodeQueue.front();
            nodeQueue.pop();
            for (const auto& child : current->children) {
                if (child) {
                    nodeQueue.push(child);
                }
            }
        }
    }
    return levelCounts;
}

// Pybind11 bindings
PYBIND11_MODULE(fastutils, m) {
    m.doc() = "C++ implementation of layerOctree.py";

    py::class_<process_info,std::shared_ptr<process_info>>(m, "process_info")
        .def(py::init<>())
        .def_readwrite("nodedict", &process_info::nodedict)
        .def_readwrite("layerIndexs", &process_info::layerIndexs)
        .def_readwrite("maxIndex", &process_info::maxIndex)
        .def_readwrite("maxLayer", &process_info::maxLayer);

    py::class_<COctree, std::shared_ptr<COctree>>(m, "COctree")
        .def(py::init<>()) // 默认构造函数
        .def(py::init<int>()) // 带层级的构造函数
        .def_readwrite("node", &COctree::node) // 绑定 node 成员变量
        .def_readwrite("level", &COctree::level); // 绑定 level 成员变量
    
    py::class_<Octant, std::shared_ptr<Octant>>(m, "Octant")
        .def(py::init<std::vector<std::shared_ptr<Octant>>, std::vector<float>, float, float, int, bool,int,int, std::vector<int>>())
        .def(py::init<const Octant&>())  // 添加深拷贝构造函数
        .def("__copy__", [](const Octant& self) {
            return std::make_shared<Octant>(self);
        })
        .def("__copy__", [](const Octant& self) {
            return std::make_shared<Octant>(self);
        })
        .def("__deepcopy__", [](const Octant& self, py::dict memo) {
            // 手动实现深拷贝，递归复制子节点
            auto copied_node = std::make_shared<Octant>(self);  // 复制当前节点的所有数据成员

            // 深度复制每一个子节点
            std::vector<std::shared_ptr<Octant>> copied_children;
            for (const auto& child : self.children) {
                if (child) {
                    copied_children.push_back(std::make_shared<Octant>(*child));  // 递归拷贝子节点
                } else {
                    copied_children.push_back(nullptr);
                }
            }

            // 更新当前节点的子节点为深拷贝后的子节点
            copied_node->children = copied_children;

            return copied_node;
        })
        .def_readwrite("children", &Octant::children)
        .def_readwrite("center", &Octant::center)
        .def_readwrite("extent", &Octant::extent)
        .def_readwrite("z_extent", &Octant::z_extent)
        .def_readwrite("depth", &Octant::depth)
        .def_readwrite("curind", &Octant::curind)
        .def_readwrite("is_leaf", &Octant::is_leaf)
        .def_readwrite("octant", &Octant::octant)
        .def_readwrite("parent", &Octant::parent)
        .def_readwrite("nodeid", &Octant::nodeid)
        .def_readwrite("point_indices", &Octant::point_indices);

    py::class_<Node,std::shared_ptr<Node>>(m, "Node")
        .def(py::init<>())
        .def_readwrite("index", &Node::index)
        .def_readwrite("fatherIndex", &Node::fatherIndex)
        .def_readwrite("dis_n_neibours", &Node::dis_n_neibours)
        .def_readwrite("location", &Node::location)
        .def_readwrite("depth", &Node::depth)
        .def_readwrite("curIndex", &Node::curIndex)
        .def_readwrite("parentOccupancy", &Node::parentOccupancy)
        .def_readwrite("occupancy", &Node::occupancy);


    m.def("octree2pointcloud", &octree2pointcloud, "Convert octree to point cloud");
    m.def("GenOctree", &GenOctree, "BFS generate octree",
        py::arg("db_np"),
        py::arg("max_layer"),
        py::arg("new_center") = std::vector<float>(3, 0.5f),
        py::arg("new_extent") = 0.5f,
        py::arg("z_rate") = 0.4f,
        py::arg("cylin") = false);

    // 绑定 NodeNumCount 函数
    m.def("NodeNumCount", &NodeNumCount, "Calculate the number of Octant nodes in the octree");
    // 绑定 NodeNumCountPerLevel 函数
    m.def("NodeNumCountPerLevel", &NodeNumCountPerLevel, "Calculate the number of Octant nodes per level in the octree");
}