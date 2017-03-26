/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#ifndef GMONOIDS_HPP_
#define GMONOIDS_HPP_

#include "util/raw-context.hpp"
#include "operators/monoids.hpp"

namespace gpu {
    class Monoid {
    public:
        virtual Value * create(RawContext * const context, 
                                Value * val_accumulating,
                                Value * val_in) = 0;

        virtual void createUpdate(RawContext * const context, 
                                    AllocaInst * val_accumulating,
                                    Value * val_in);

        virtual void createAtomicUpdate(RawContext * const context, 
                                        Value * accumulator_ptr, 
                                        Value * val_in, 
                                        llvm::AtomicOrdering order = 
                                            llvm::AtomicOrdering::Monotonic)=0;

        virtual Value * createWarpAggregateToAll(RawContext * const context, 
                                            Value * val_in);

        virtual Value * createWarpAggregateTo0(RawContext * const context, 
                                            Value * val_in){
            return createWarpAggregateToAll(context, val_in);
        }

        static Monoid * get(::Monoid m);
    };

    class MaxMonoid : public Monoid{
        Value * create(RawContext * const context, 
                        Value * val_accumulating,
                        Value * val_in);

        void createUpdate(RawContext * const context, 
                            AllocaInst * val_accumulating,
                            Value * val_in);

        void createAtomicUpdate(RawContext * const context, 
                                Value * accumulator_ptr, 
                                Value * val_in, 
                                llvm::AtomicOrdering order = 
                                            llvm::AtomicOrdering::Monotonic);
    };

    class SumMonoid : public Monoid{
        Value * create(RawContext * const context, 
                        Value * val_accumulating,
                        Value * val_in);

        void createAtomicUpdate(RawContext * const context, 
                                Value * accumulator_ptr, 
                                Value * val_in, 
                                llvm::AtomicOrdering order = 
                                            llvm::AtomicOrdering::Monotonic);
    };

    class LogOrMonoid : public Monoid{
        Value * create(RawContext * const context, 
                        Value * val_accumulating,
                        Value * val_in);

        void createAtomicUpdate(RawContext * const context, 
                                Value * accumulator_ptr, 
                                Value * val_in, 
                                llvm::AtomicOrdering order = 
                                            llvm::AtomicOrdering::Monotonic);

        Value * createWarpAggregateToAll(RawContext * const context, 
                                            Value * val_in);
    };

    class LogAndMonoid : public Monoid{
        Value * create(RawContext * const context, 
                        Value * val_accumulating,
                        Value * val_in);

        void createAtomicUpdate(RawContext * const context, 
                                Value *accumulator_ptr, 
                                Value * val_in, 
                                llvm::AtomicOrdering order = 
                                            llvm::AtomicOrdering::Monotonic);
        
        Value * createWarpAggregateToAll(RawContext * const context, 
                                            Value * val_in);
    };

    class BitOrMonoid : public Monoid{
        Value * create(RawContext * const context, 
                        Value * val_accumulating,
                        Value * val_in);

        void createAtomicUpdate(RawContext * const context, 
                                Value * accumulator_ptr, 
                                Value * val_in, 
                                llvm::AtomicOrdering order = 
                                            llvm::AtomicOrdering::Monotonic);
    };

    class BitAndMonoid : public Monoid{
        Value * create(RawContext * const context, 
                        Value * al_accumulating,
                        Value * val_in);

        void createAtomicUpdate(RawContext * const context, 
                                Value * accumulator_ptr, 
                                Value * val_in, 
                                llvm::AtomicOrdering order = 
                                            llvm::AtomicOrdering::Monotonic);
    };

}

#endif /* GMONOIDS_HPP_ */
