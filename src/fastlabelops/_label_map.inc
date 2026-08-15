namespace label_map_impl {

constexpr std::size_t kDefaultInitialCapacity = 16384;
constexpr std::size_t kLoadNumerator = 7;
constexpr std::size_t kLoadDenominator = 10;

inline std::size_t growth_threshold(std::size_t bucket_count) noexcept {
    return (bucket_count * kLoadNumerator) / kLoadDenominator;
}

inline std::size_t bucket_count_for_expected_entries(std::size_t expected_entries) {
    std::size_t bucket_count = 16;
    while (growth_threshold(bucket_count) < expected_entries) {
        if (bucket_count > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::bad_alloc();
        }
        bucket_count <<= 1;
    }
    return bucket_count;
}

template <typename Key, typename Value>
struct Entry {
    Key key = Key{0};
    Value value = Value{};
};

// Label 0 is reserved as the empty-bucket sentinel. All callers skip
// background labels before insertion.
template <typename Key, typename Value>
class LabelMap {
public:
    static_assert(std::is_unsigned_v<Key>);

    explicit LabelMap(std::size_t expected_entries = kDefaultInitialCapacity) {
        const std::size_t cap = bucket_count_for_expected_entries(expected_entries);
        entries_.resize(cap);
        configure_capacity(cap);
    }

    inline std::pair<Value*, bool> get_or_insert(Key key, Value initial) {
        assert(key != Key{0});
        std::size_t i = index_for(key);
        for (;;) {
            Entry<Key, Value>& e = entries_[i];
            if (e.key == key) return {&e.value, false};
            if (e.key == Key{0}) {
                if (size_ >= grow_at_) {
                    grow();
                    return get_or_insert(key, initial);
                }
                e.key = key;
                e.value = initial;
                ++size_;
                return {&e.value, true};
            }
            i = (i + 1) & mask_;
        }
    }

    inline const Value* find(Key key) const noexcept {
        assert(key != Key{0});
        std::size_t i = index_for(key);
        for (;;) {
            const Entry<Key, Value>& e = entries_[i];
            if (e.key == key) return &e.value;
            if (e.key == Key{0}) return nullptr;
            i = (i + 1) & mask_;
        }
    }

    template <typename Func>
    void for_each(Func&& func) const {
        for (const Entry<Key, Value>& entry : entries_) {
            if (entry.key != Key{0}) func(entry.key, entry.value);
        }
    }

private:
    inline std::size_t index_for(Key key) const noexcept {
        constexpr uint64_t kFib = 11400714819323198485ULL;
        const uint64_t hash = static_cast<uint64_t>(key) * kFib;
        return static_cast<std::size_t>(hash >> shift_);
    }

    inline void insert_no_grow(Key key, Value value) noexcept {
        std::size_t i = index_for(key);
        while (entries_[i].key != Key{0}) i = (i + 1) & mask_;
        entries_[i].key = key;
        entries_[i].value = value;
        ++size_;
    }

    void configure_capacity(std::size_t cap) noexcept {
        mask_ = cap - 1;
        unsigned bits = 0;
        for (std::size_t x = cap; x > 1; x >>= 1) ++bits;
        shift_ = 64u - bits;
        grow_at_ = growth_threshold(cap);
    }

    void grow() {
        if (entries_.size() > entries_.max_size() / 2) throw std::bad_alloc();
        std::vector<Entry<Key, Value>> old = std::move(entries_);
        const std::size_t new_cap = old.size() * 2;
        entries_.clear();
        entries_.resize(new_cap);
        configure_capacity(new_cap);
        size_ = 0;
        for (const Entry<Key, Value>& e : old) {
            if (e.key != Key{0}) insert_no_grow(e.key, e.value);
        }
    }

    std::vector<Entry<Key, Value>> entries_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
    std::size_t grow_at_ = 0;
    unsigned shift_ = 0;
};

}  // namespace label_map_impl
