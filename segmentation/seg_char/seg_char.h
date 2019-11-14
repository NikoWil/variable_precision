//
// Created by niko on 9/13/19.
//

#ifndef CODE_SEG_CHAR_H
#define CODE_SEG_CHAR_H

#include <cstdint>
#include <iostream>

namespace {
union Segmentation {
  static_assert(
      sizeof(double) == 8,
      "Assumption of IEEE floating point necessiates 8 bytes per double");
  double d;
  unsigned char c[sizeof(double)];
  uint64_t u;
};

template <int start, int end> struct Extract_helper {
  static_assert(start >= 0, "Extracting bytes before start of double");
  static_assert(end < sizeof(double), "Extracting bytes beyond end of double");
  static_assert(start < end, "Only forward copying is supported");

  static constexpr void copy(double d, unsigned char *c) {
    Segmentation seg{d};
    *c = seg.c[sizeof(double) - start - 1];
    Extract_helper<start + 1, end> eh;
    eh.copy(d, c + 1);
  }
};

template <int index> struct Extract_helper<index, index> {
  static_assert(index >= 0, "Extracting bytes before end of double");
  static_assert(index < sizeof(double),
                "Extracting bytes beyond end of double");

  static constexpr void copy(double d, unsigned char *c) {
    Segmentation seg{d};
    *c = seg.c[sizeof(double) - index - 1];
  }
};

template <int start, int end>
constexpr void extract_slice(double d,
                             unsigned char (&chars)[end - start + 1]) {
  static_assert(start >= 0, "Extracting bytes before start of double");
  static_assert(end < sizeof(double), "Extracting bytes beyond end of double");
  static_assert(start <= end, "Extracting negative sized double slice");

  Extract_helper<start, end> eh;
  eh.copy(d, chars);
}

template <int start, int end> struct Insert_helper {
  static_assert(start >= 0, "Inserting bytes before start of double");
  static_assert(end < sizeof(double), "Inserting bytes beyond end of double");
  static_assert(start < end, "Inserting negative sized slice");

  static constexpr uint64_t insert(const unsigned char *bytes) {
    Segmentation seg{0.};
    seg.c[sizeof(double) - start - 1] = *bytes;
    Insert_helper<start + 1, end> ih;
    return seg.u | ih.insert(bytes + 1);
  }
};

template <int index> struct Insert_helper<index, index> {
  static_assert(index >= 0, "Inserting bytes before start of double");
  static_assert(index < sizeof(double), "Inserting bytes beyond end of double");

  static constexpr uint64_t insert(const unsigned char *bytes) {
    Segmentation seg{0.};
    seg.c[sizeof(double) - index - 1] = *bytes;
    return seg.u;
  }
};

template <int start, int end>
constexpr double insert_slice(const unsigned char (&chars)[end - start + 1]) {
  static_assert(start >= 0, "Inserting bytes before start of double");
  static_assert(end < sizeof(double), "Inserting bytes beyond end of double");
  static_assert(start <= end, "Inserting negative sized or empty slice");

  Insert_helper<start, end> ih;
  uint64_t u = ih.insert(chars);
  Segmentation s{0.};
  s.u = u;
  return s.d;
}

template <int index, int remaining> struct Compare_helper {
  static constexpr int compare(const unsigned char *c1,
                               const unsigned char *c2) {
    if (*c1 != *c2) {
      return index;
    } else {
      return Compare_helper<index + 1, remaining - 1>::compare(c1 + 1, c2 + 1);
    }
  }
};

template <int index> struct Compare_helper<index, 0> {
  static constexpr int compare(const unsigned char *c1,
                               const unsigned char *c2) {
    (void)c1;
    (void)c2;
    return index;
  }
};
}

namespace seg {
template <int start, int end> struct Double_slice {
  static_assert(start >= 0, "Slice started before start of double");
  static_assert(end < sizeof(double), "Slice exceeding end of double");

  explicit constexpr Double_slice() : Double_slice(0.) {}

  explicit constexpr Double_slice(double d) {
    extract_slice<start, end>(d, bytes);
  }

  template <int new_end>
  explicit constexpr Double_slice(Double_slice<start, new_end> ds)
      : Double_slice(ds.to_double()) {
    static_assert(new_end <= end,
                  "seg::Double_slice lossy construction of lesser precision slice from higher precision slice");
  }

  constexpr double to_double() const { return insert_slice<start, end>(bytes); }

  constexpr int compare_bytes(Double_slice<start, end> other) const {
    return Compare_helper<0, end - start + 1>::compare(bytes, other.bytes);
  }

  constexpr unsigned char *get_bytes() { return bytes; }

  void print_bytes() const {
    for (auto b : bytes) {
      std::cout << (+b & 0xFF) << " ";
    }
    std::cout << std::endl;
  }

private:
  unsigned char bytes[end - start + 1];
};

// Implement compare_bytes - how, without segmentation?
template <> struct Double_slice<0, sizeof(double) - 1> {
public:
  explicit constexpr Double_slice() : Double_slice(0.) {}

  explicit constexpr Double_slice(double d) : d{d} {}

  template <int new_end>
  explicit constexpr Double_slice(Double_slice<0, new_end> ds) : Double_slice(ds.to_double()) {}

  constexpr double to_double() const { return d; }

  constexpr int compare_bytes(Double_slice<0, sizeof(double) - 1> other) const {
    unsigned char bytes_1[sizeof(double)]{};
    unsigned char bytes_2[sizeof(double)]{};
    extract_slice<0, sizeof(double) - 1>(d, bytes_1);
    extract_slice<0, sizeof(double) - 1>(other.d, bytes_2);

    return Compare_helper<0, sizeof(double)>::compare(bytes_1, bytes_2);
  }

  void print_bytes() const {
    Segmentation seg{d};
    for (auto c : seg.c) {
      std::cout << (c & 0xFFu) << " ";
    }
    std::cout << std::endl;
  }

private:
  double d;
};
}
#endif // CODE_SEG_CHAR_H
