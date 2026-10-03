#include "minicompiler/parser.hpp"

#include "minicompiler/random.hpp"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace minicompiler {
namespace {

enum class TokKind { Ident, Number, Punct };

struct Token {
	TokKind kind;
	std::string text;
};

bool is_digit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }
bool is_ident_start(char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_'; }
bool is_ident_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_'; }

// Splits one line into identifiers, numbers and punctuation. `#` starts a
// comment that runs to the end of the line.
Result<std::vector<Token>> tokenize(std::string_view line) {
	std::vector<Token> tokens;
	std::size_t i = 0;
	while (i < line.size()) {
		const char c = line[i];
		if (std::isspace(static_cast<unsigned char>(c))) {
			++i;
			continue;
		}
		if (c == '#') break;
		if (is_ident_start(c)) {
			std::size_t j = i + 1;
			while (j < line.size() && is_ident_char(line[j])) ++j;
			tokens.push_back({TokKind::Ident, std::string(line.substr(i, j - i))});
			i = j;
			continue;
		}
		const bool signed_number = (c == '-' || c == '+') && i + 1 < line.size() &&
		                           (is_digit(line[i + 1]) || line[i + 1] == '.');
		if (is_digit(c) || c == '.' || signed_number) {
			std::size_t j = i + 1;
			while (j < line.size()) {
				if (is_digit(line[j]) || line[j] == '.') {
					++j;
				} else if (line[j] == 'e' || line[j] == 'E') {
					++j;
					if (j < line.size() && (line[j] == '+' || line[j] == '-')) ++j;
				} else {
					break;
				}
			}
			tokens.push_back({TokKind::Number, std::string(line.substr(i, j - i))});
			i = j;
			continue;
		}
		if (std::string_view(":=,[]()").find(c) != std::string_view::npos) {
			tokens.push_back({TokKind::Punct, std::string(1, c)});
			++i;
			continue;
		}
		return Error{std::string("unexpected character '") + c + "'"};
	}
	return tokens;
}

// Cursor over the tokens of one statement.
class Cursor {
public:
	explicit Cursor(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

	bool at_end() const { return pos_ >= tokens_.size(); }
	const Token& peek() const { return tokens_[pos_]; }

	bool next_is(TokKind kind) const { return !at_end() && peek().kind == kind; }
	bool next_is_punct(char c) const { return next_is(TokKind::Punct) && peek().text[0] == c; }

	bool accept_punct(char c) {
		if (!next_is_punct(c)) return false;
		++pos_;
		return true;
	}

	Status expect_punct(char c) {
		if (accept_punct(c)) return Status();
		return Error{std::string("expected '") + c + "' " + found()};
	}

	Result<std::string> ident(const char* what) {
		if (!next_is(TokKind::Ident)) return Error{std::string("expected ") + what + " " + found()};
		return tokens_[pos_++].text;
	}

	Result<double> number(const char* what) {
		if (!next_is(TokKind::Number)) return Error{std::string("expected ") + what + " " + found()};
		const std::string& text = tokens_[pos_++].text;
		char* end = nullptr;
		errno = 0;
		const double v = std::strtod(text.c_str(), &end);
		if (end != text.c_str() + text.size() || errno == ERANGE) return Error{"bad number '" + text + "'"};
		return v;
	}

	Result<std::int64_t> integer(const char* what) {
		if (!next_is(TokKind::Number)) return Error{std::string("expected ") + what + " " + found()};
		const std::string& text = tokens_[pos_++].text;
		char* end = nullptr;
		errno = 0;
		const long long v = std::strtoll(text.c_str(), &end, 10);
		if (end != text.c_str() + text.size() || errno == ERANGE) return Error{"expected an integer, got '" + text + "'"};
		return static_cast<std::int64_t>(v);
	}

	std::string found() const { return at_end() ? "at end of line" : "but found '" + peek().text + "'"; }

private:
	std::vector<Token> tokens_;
	std::size_t pos_ = 0;
};

class GraphParser {
public:
	explicit GraphParser(const ParseOptions& options) : options_(options) {}

	Result<Graph> parse(std::string_view text);

private:
	Status statement(Cursor& c);
	Status parse_graph_name(Cursor& c);
	Status parse_dim(Cursor& c);
	Status parse_input(Cursor& c);
	Status parse_const(Cursor& c);
	Status parse_output(Cursor& c);
	Status parse_op(Cursor& c, const std::string& name);
	Result<TensorType> parse_type(Cursor& c);
	Result<std::vector<float>> parse_init(Cursor& c, const TensorType& type);
	Result<NodeId> lookup(const std::string& name) const;
	Status define(const std::string& name, Result<NodeId> id);

	const ParseOptions& options_;
	Graph graph_;
	std::map<std::string, std::int64_t> dims_;
	std::unordered_map<std::string, NodeId> values_;
};

Result<Graph> GraphParser::parse(std::string_view text) {
	std::size_t line_no = 0;
	std::size_t start = 0;
	while (start <= text.size()) {
		std::size_t end = text.find('\n', start);
		if (end == std::string_view::npos) end = text.size();
		std::string_view line = text.substr(start, end - start);
		if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
		++line_no;

		Result<std::vector<Token>> tokens = tokenize(line);
		Status st = tokens.status();
		if (st.ok() && !tokens->empty()) {
			Cursor c(std::move(tokens).value());
			st = statement(c);
		}
		if (!st.ok()) return Error{"line " + std::to_string(line_no) + ": " + st.message()};
		if (end == text.size()) break;
		start = end + 1;
	}
	for (const auto& [name, value] : options_.dims) {
		(void)value;
		if (dims_.count(name) == 0) return Error{"the graph declares no dim '" + name + "'"};
	}
	if (graph_.outputs().empty()) return Error{"the graph declares no outputs"};
	Status st = graph_.verify();
	if (!st.ok()) return st.error();
	return std::move(graph_);
}

Status GraphParser::statement(Cursor& c) {
	Result<std::string> first = c.ident("a statement");
	if (!first.ok()) return first.error();
	const std::string& word = first.value();
	Status st;
	if (word == "graph") {
		st = parse_graph_name(c);
	} else if (word == "dim") {
		st = parse_dim(c);
	} else if (word == "input") {
		st = parse_input(c);
	} else if (word == "const") {
		st = parse_const(c);
	} else if (word == "output") {
		st = parse_output(c);
	} else {
		st = parse_op(c, word);
	}
	if (!st.ok()) return st;
	if (!c.at_end()) return Error{"unexpected '" + c.peek().text + "'"};
	return Status();
}

Status GraphParser::parse_graph_name(Cursor& c) {
	Result<std::string> name = c.ident("a graph name");
	if (!name.ok()) return name.error();
	if (graph_.num_nodes() > 0) return Error{"'graph' must come before any node"};
	graph_ = Graph(name.value());
	return Status();
}

Status GraphParser::parse_dim(Cursor& c) {
	Result<std::string> name = c.ident("a dim name");
	if (!name.ok()) return name.error();
	Status st = c.expect_punct('=');
	if (!st.ok()) return st;
	Result<std::int64_t> value = c.integer("a dim value");
	if (!value.ok()) return value.error();
	if (dims_.count(name.value()) != 0) return Error{"dim '" + name.value() + "' is declared twice"};
	auto override_it = options_.dims.find(name.value());
	const std::int64_t v = override_it != options_.dims.end() ? override_it->second : value.value();
	if (v < 1) return Error{"dim '" + name.value() + "' must be >= 1, got " + std::to_string(v)};
	dims_[name.value()] = v;
	return Status();
}

Status GraphParser::parse_input(Cursor& c) {
	Result<std::string> name = c.ident("an input name");
	if (!name.ok()) return name.error();
	Status st = c.expect_punct(':');
	if (!st.ok()) return st;
	Result<TensorType> type = parse_type(c);
	if (!type.ok()) return type.error();
	return define(name.value(), graph_.add_input(name.value(), std::move(type).value()));
}

Status GraphParser::parse_const(Cursor& c) {
	Result<std::string> name = c.ident("a constant name");
	if (!name.ok()) return name.error();
	Status st = c.expect_punct(':');
	if (!st.ok()) return st;
	Result<TensorType> type = parse_type(c);
	if (!type.ok()) return type.error();
	st = c.expect_punct('=');
	if (!st.ok()) return st;
	Result<std::vector<float>> values = parse_init(c, type.value());
	if (!values.ok()) return values.error();
	return define(name.value(),
	              graph_.add_constant(name.value(), std::move(type).value(), std::move(values).value()));
}

Status GraphParser::parse_output(Cursor& c) {
	do {
		Result<std::string> name = c.ident("an output name");
		if (!name.ok()) return name.error();
		Result<NodeId> id = lookup(name.value());
		if (!id.ok()) return id.error();
		Status st = graph_.add_output(id.value());
		if (!st.ok()) return st;
	} while (c.accept_punct(','));
	return Status();
}

Status GraphParser::parse_op(Cursor& c, const std::string& name) {
	Status st = c.expect_punct('=');
	if (!st.ok()) return Error{"unknown statement '" + name + "'"};
	Result<std::string> op_word = c.ident("an op name");
	if (!op_word.ok()) return op_word.error();
	std::optional<OpKind> op = op_from_name(op_word.value());
	if (!op) return Error{"unknown op '" + op_word.value() + "'"};

	std::vector<NodeId> operands;
	do {
		Result<std::string> operand = c.ident("an operand");
		if (!operand.ok()) return operand.error();
		Result<NodeId> id = lookup(operand.value());
		if (!id.ok()) return id.error();
		operands.push_back(id.value());
	} while (c.accept_punct(','));
	return define(name, graph_.add_op(*op, std::move(operands), name));
}

Result<TensorType> GraphParser::parse_type(Cursor& c) {
	Result<std::string> dtype = c.ident("an element type");
	if (!dtype.ok()) return dtype.error();
	if (dtype.value() != "f32") return Error{"unsupported element type '" + dtype.value() + "' (only f32)"};
	Status st = c.expect_punct('[');
	if (!st.ok()) return st.error();
	Shape shape;
	if (c.accept_punct(']')) return TensorType(shape);
	do {
		if (c.next_is(TokKind::Ident)) {
			const std::string dim = c.ident("a dim").value();
			auto it = dims_.find(dim);
			if (it == dims_.end()) return Error{"unknown dim '" + dim + "'"};
			shape.push_back(it->second);
		} else {
			Result<std::int64_t> d = c.integer("a dimension");
			if (!d.ok()) return d.error();
			shape.push_back(d.value());
		}
	} while (c.accept_punct(','));
	st = c.expect_punct(']');
	if (!st.ok()) return st.error();
	// Checked here, before a constant's initializer allocates its values.
	if (!checked_num_elements(shape)) {
		return Error{"invalid shape " + to_string(shape) +
		             ": dimensions must be >= 1 and the total at most 2^31 - 1 elements"};
	}
	return TensorType(shape);
}

Result<std::vector<float>> GraphParser::parse_init(Cursor& c, const TensorType& type) {
	const std::size_t n = type.num_elements();
	if (c.next_is(TokKind::Number)) {
		const double v = c.number("a value").value();
		return std::vector<float>(n, static_cast<float>(v));
	}
	if (c.accept_punct('[')) {
		std::vector<float> values;
		do {
			Result<double> v = c.number("a value");
			if (!v.ok()) return v.error();
			values.push_back(static_cast<float>(v.value()));
		} while (c.accept_punct(','));
		Status st = c.expect_punct(']');
		if (!st.ok()) return st.error();
		if (values.size() != n) {
			return Error{"expected " + std::to_string(n) + " values for " + to_string(type) + ", got " +
			             std::to_string(values.size())};
		}
		return values;
	}
	Result<std::string> fn = c.ident("a value, [values] or uniform(...)");
	if (!fn.ok()) return fn.error();
	if (fn.value() != "uniform") return Error{"unknown initializer '" + fn.value() + "'"};
	Status st = c.expect_punct('(');
	if (!st.ok()) return st.error();
	std::optional<std::uint64_t> seed;
	std::map<std::string, double> bounds;
	do {
		Result<std::string> key = c.ident("seed, lo or hi");
		if (!key.ok()) return key.error();
		if (key.value() != "seed" && key.value() != "lo" && key.value() != "hi") {
			return Error{"uniform() takes seed, lo and hi, not '" + key.value() + "'"};
		}
		st = c.expect_punct('=');
		if (!st.ok()) return st.error();
		if (key.value() == "seed") {
			Result<std::int64_t> s = c.integer("a seed");
			if (!s.ok()) return s.error();
			if (s.value() < 0) return Error{"seed must be >= 0"};
			seed = static_cast<std::uint64_t>(s.value());  // kept as an integer: a double loses bits above 2^53
		} else {
			Result<double> v = c.number("a bound");
			if (!v.ok()) return v.error();
			bounds[key.value()] = v.value();
		}
	} while (c.accept_punct(','));
	st = c.expect_punct(')');
	if (!st.ok()) return st.error();
	if (!seed || bounds.size() != 2) return Error{"uniform() needs seed, lo and hi"};
	const float lo = static_cast<float>(bounds["lo"]);
	const float hi = static_cast<float>(bounds["hi"]);
	if (!(lo < hi)) return Error{"uniform() needs lo < hi"};
	return uniform_values(n, *seed, lo, hi);
}

Result<NodeId> GraphParser::lookup(const std::string& name) const {
	auto it = values_.find(name);
	if (it == values_.end()) return Error{"unknown value '" + name + "'"};
	return it->second;
}

Status GraphParser::define(const std::string& name, Result<NodeId> id) {
	if (!id.ok()) return id.error();
	if (!values_.emplace(name, id.value()).second) return Error{"'" + name + "' is defined twice"};
	return Status();
}

}

Result<Graph> parse_graph(std::string_view text, const ParseOptions& options) {
	return GraphParser(options).parse(text);
}

Result<Graph> parse_graph_file(const std::string& path, const ParseOptions& options) {
	std::ifstream in(path, std::ios::binary);
	if (!in) return Error{"cannot open '" + path + "'"};
	std::ostringstream text;
	text << in.rdbuf();
	Result<Graph> g = parse_graph(text.str(), options);
	if (!g.ok()) return Error{path + ": " + g.error().message};
	return g;
}

std::string describe_constant(const Node& constant) {
	const std::vector<float>& v = *constant.constant;
	std::ostringstream os;
	if (v.size() == 1) {
		os << v[0];
	} else if (v.size() <= 4) {
		os << "[";
		for (std::size_t i=0; i<v.size(); ++i) os << (i ? ", " : "") << v[i];
		os << "]";
	} else {
		os << "<" << v.size() << " values>";
	}
	return os.str();
}

std::string print_graph(const Graph& graph) {
	std::ostringstream os;
	os << "graph " << graph.name() << "\n";
	for (std::size_t i=0; i<graph.num_nodes(); ++i) {
		const NodeId id = static_cast<NodeId>(i);
		const Node& n = graph.node(id);
		os << "  %" << id << " " << n.name << " = ";
		if (n.op == OpKind::Constant) {
			os << "const " << describe_constant(n);
		} else {
			os << op_name(n.op);
			for (std::size_t k=0; k<n.inputs.size(); ++k) os << (k ? ", %" : " %") << n.inputs[k];
		}
		os << " : " << to_string(n.type);
		if (graph.is_output(id)) os << "  (output)";
		os << "\n";
		if (n.op == OpKind::FusedElementwise) {
			std::istringstream body(to_string(*n.fused));
			for (std::string line; std::getline(body, line);) os << "        " << line << "\n";
		}
	}
	return os.str();
}

}
