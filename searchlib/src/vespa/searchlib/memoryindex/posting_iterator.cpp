// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "posting_iterator.h"
#include <vespa/searchlib/queryeval/iterators.h>
#include <vespa/vespalib/btree/btreeiterator.hpp>
#include <vespa/vespalib/btree/btreenode.hpp>
#include <vespa/vespalib/btree/btreenodeallocator.hpp>
#include <vespa/vespalib/btree/btreenodestore.hpp>
#include <vespa/vespalib/btree/btreeroot.hpp>
#include <vespa/vespalib/btree/btreestore.hpp>

#include <vespa/log/log.h>
LOG_SETUP(".searchlib.memoryindex.posting_iterator");

namespace search::memoryindex {

/**
 * Base search iterator over memory field index posting list.
 *
 * The template parameter specifies whether the wrapped posting list has interleaved features or not.
 */
template <bool interleaved_features>
class PostingIteratorBase : public queryeval::RankedSearchIteratorBase {
protected:
    using FieldIndexType = FieldIndex<interleaved_features>;
    using PostingListIteratorType = typename FieldIndexType::PostingList::ConstIterator;
    PostingListIteratorType _itr;
    const FeatureStore& _feature_store;
    FeatureStore::DecodeContextCooked _feature_decoder;

public:
    PostingIteratorBase(PostingListIteratorType itr,
                        const FeatureStore& feature_store,
                        uint32_t field_id,
                        const fef::TermFieldMatchDataArray& match_data);
    ~PostingIteratorBase();

    void doSeek(uint32_t docId) override;
    void initRange(uint32_t begin, uint32_t end) override;
    Trinary is_strict() const override { return Trinary::True; }
};

template <bool interleaved_features>
PostingIteratorBase<interleaved_features>::PostingIteratorBase(PostingListIteratorType itr,
                                                               const FeatureStore& feature_store,
                                                               uint32_t field_id,
                                                               const fef::TermFieldMatchDataArray& match_data) :
    queryeval::RankedSearchIteratorBase(match_data),
    _itr(itr),
    _feature_store(feature_store),
    _feature_decoder(nullptr)
{
    _feature_store.setupForField(field_id, _feature_decoder);
}

template <bool interleaved_features>
PostingIteratorBase<interleaved_features>::~PostingIteratorBase() = default;

template <bool interleaved_features>
void
PostingIteratorBase<interleaved_features>::initRange(uint32_t begin, uint32_t end)
{
    SearchIterator::initRange(begin, end);
    _itr.lower_bound(begin);
    if (!_itr.valid() || isAtEnd(_itr.getKey())) {
        setAtEnd();
    } else {
        setDocId(_itr.getKey());
    }
    clearUnpacked();
}

template <bool interleaved_features>
void
PostingIteratorBase<interleaved_features>::doSeek(uint32_t docId)
{
    if (getUnpacked()) {
        clearUnpacked();
    }
    _itr.linearSeek(docId);
    if (!_itr.valid()) {
        setAtEnd();
    } else {
        setDocId(_itr.getKey());
    }
}

/**
 * Search iterator over memory field index posting list.
 *
 * Template parameters:
 *   - interleaved_features: specifies whether the wrapped posting list has interleaved features or not.
 *   - unpack_normal_features: specifies whether to unpack normal features or not.
 *   - unpack_interleaved_features: specifies whether to unpack interleaved features or not.
 */
template <bool interleaved_features, bool unpack_normal_features, bool unpack_interleaved_features>
class PostingIterator : public PostingIteratorBase<interleaved_features> {
public:
    using ParentType = PostingIteratorBase<interleaved_features>;

    using ParentType::ParentType;
    using ParentType::_feature_decoder;
    using ParentType::_feature_store;
    using ParentType::_itr;
    using ParentType::_matchData;
    using ParentType::getDocId;
    using ParentType::getUnpacked;
    using ParentType::setUnpacked;

    void doUnpack(uint32_t docId) override;
};

template <bool interleaved_features, bool unpack_normal_features, bool unpack_interleaved_features>
void
PostingIterator<interleaved_features, unpack_normal_features, unpack_interleaved_features>::doUnpack(uint32_t docId)
{
    if (!_matchData.valid() || getUnpacked()) {
        return;
    }
    assert(docId == getDocId());
    assert(_itr.valid());
    assert(docId == _itr.getKey());
    if (unpack_normal_features) {
        datastore::EntryRef featureRef(_itr.getData().get_features());
        _feature_store.setupForUnpackFeatures(featureRef, _feature_decoder);
        _feature_decoder.unpackFeatures(_matchData, docId);
        setUnpacked();
    }
    if (unpack_interleaved_features) {
        // TODO: implement
    }
}

template <bool interleaved_features>
queryeval::SearchIterator::UP
make_search_iterator(typename FieldIndex<interleaved_features>::PostingList::ConstIterator itr,
                     const FeatureStore& feature_store,
                     uint32_t field_id,
                     const fef::TermFieldMatchDataArray& match_data)
{
    assert(match_data.size() == 1);
    return std::make_unique<PostingIterator<interleaved_features, true, false>>
            (itr, feature_store, field_id, match_data);
}

template
queryeval::SearchIterator::UP
make_search_iterator<false>(typename FieldIndex<false>::PostingList::ConstIterator,
                            const FeatureStore&,
                            uint32_t,
                            const fef::TermFieldMatchDataArray&);

template class PostingIteratorBase<false>;

template class PostingIterator<false, true, false>;

}

namespace search::btree {

template class BTreeNodeTT<uint32_t,
                           search::memoryindex::PostingListEntry<false>,
                           search::btree::NoAggregated,
                           BTreeDefaultTraits::INTERNAL_SLOTS>;

template class BTreeLeafNode<uint32_t,
                             search::memoryindex::PostingListEntry<false>,
                             search::btree::NoAggregated,
                             BTreeDefaultTraits::LEAF_SLOTS>;

template class BTreeNodeStore<uint32_t,
                              search::memoryindex::PostingListEntry<false>,
                              search::btree::NoAggregated,
                              BTreeDefaultTraits::INTERNAL_SLOTS,
                              BTreeDefaultTraits::LEAF_SLOTS>;

template class BTreeIteratorBase<uint32_t,
                                 search::memoryindex::PostingListEntry<false>,
                                 search::btree::NoAggregated,
                                 BTreeDefaultTraits::INTERNAL_SLOTS,
                                 BTreeDefaultTraits::LEAF_SLOTS,
                                 BTreeDefaultTraits::PATH_SIZE>;

template class BTreeIterator<uint32_t,
                             search::memoryindex::PostingListEntry<false>,
                             search::btree::NoAggregated,
                             std::less<uint32_t>,
                             BTreeDefaultTraits>;

template class BTree<uint32_t,
                     search::memoryindex::PostingListEntry<false>,
                     search::btree::NoAggregated,
                     std::less<uint32_t>,
                     BTreeDefaultTraits>;

template class BTreeRoot<uint32_t,
                         search::memoryindex::PostingListEntry<false>,
                         search::btree::NoAggregated,
                         std::less<uint32_t>,
                         BTreeDefaultTraits>;

template class BTreeRootBase<uint32_t,
                             search::memoryindex::PostingListEntry<false>,
                             search::btree::NoAggregated,
                             BTreeDefaultTraits::INTERNAL_SLOTS,
                             BTreeDefaultTraits::LEAF_SLOTS>;

template class BTreeNodeAllocator<uint32_t,
                                  search::memoryindex::PostingListEntry<false>,
                                  search::btree::NoAggregated,
                                  BTreeDefaultTraits::INTERNAL_SLOTS,
                                  BTreeDefaultTraits::LEAF_SLOTS>;

}
