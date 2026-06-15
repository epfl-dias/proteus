#ifndef DEBUG_HPP
#define DEBUG_HPP

#ifndef FLS_DEBUG_COLOR_HPP
#define FLS_DEBUG_COLOR_HPP

#include <cstdint>
#include <ostream>

namespace fastlanes::debug {
enum Code : uint32_t {
	FG_BLACK   = 30,
	FG_RED     = 31,
	FG_GREEN   = 32,
	FG_YELLOW  = 33,
	FG_BLUE    = 34,
	FG_MAGENTA = 35,
	FG_CYAN    = 36,
	FG_WHITE   = 37,
	FG_DEFAULT = 39,
	BG_RED     = 41,
	BG_GREEN   = 42,
	BG_BLUE    = 44,
	BG_DEFAULT = 49

};

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& reset(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_BLACK << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& black(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_BLACK << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& bold_black(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[1m\033[" << FG_BLACK << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& bold_blue(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[1m\033[" << FG_BLUE << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& red(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_RED << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& magenta(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_MAGENTA << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& yellow(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_YELLOW << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& def(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_DEFAULT << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& green(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_GREEN << "m";
}
} // namespace fastlanes::debug
#endif // FLS_DEBUG_COLOR_HPP

#define FLS_SHOW(a)                                                                                                    \
	std::cout << fastlanes::debug::yellow << "-- " << #a << ": " << (a) << fastlanes::debug::def << '\n';
#define FLS_LOG(m)     std::cout << fastlanes::debug::yellow << "-- " << m << fastlanes::debug::def << '\n';
#define FLS_CERR(a)    std::cout << fastlanes::debug::red << "-- " << #a << ": " << (a) << fastlanes::debug::def << '\n';
#define FLS_SUCCESS(m) std::cout << fastlanes::debug::green << "-- " << m << fastlanes::debug::def << '\n';
#define FLS_RESULT(m)  std::cout << fastlanes::debug::bold_blue << "-- " << m << fastlanes::debug::def << '\n';

template <typename T>
void PRINT(T* arr, const char* str) {
	printf("\n ==================   %s   ================= \n ", str);

	for (int ITEM = 0; ITEM < 1024; ++ITEM) {
		if (ITEM % 128 == 0) { printf("\n"); }
		printf(" %d | ", arr[ITEM]);
	}

	printf("\n");
}

#endif // DEBUG_HPP
