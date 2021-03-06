#pragma once

#include <memory>

#include "../graph.h"
#include "../helpers.h"

namespace ad {
namespace nn {

struct DiscreteMRNNParams {
    std::vector<std::shared_ptr<Param>> whx_;
    std::vector<std::shared_ptr<Param>> whh_;
    std::shared_ptr<Param> bh_;
    std::shared_ptr<Param> h_;

    DiscreteMRNNParams(int out_sz, size_t in_sz);

    void ResizeInput(size_t size, double init = 1);
    void ResizeOutput(size_t size, double init = 1);

    void Serialize(std::ostream& out) const;
    static DiscreteMRNNParams FromSerialized(std::istream& in);
};

class DiscreteMRNNLayer {
    private:
        std::vector<Var> whx_;
        std::vector<Var> whh_;
        Var bh_;
        Var h_;

        std::vector<Var> used_vars_;

    public:
        DiscreteMRNNLayer(
                ComputationGraph& g,
                const DiscreteMRNNParams& params,
                bool learnable = true);

        void SetHidden(Var h) { h_ = h; }
        Var GetHidden() const { return h_; }

        Var Step(int x);
        std::vector<Var> Params() const;
};


} // nn
} // ad
