#include "minicompiler/viz/dot_export.hpp"

#include "minicompiler/parser.hpp"

#include <fstream>
#include <sstream>

namespace minicompiler {
namespace {

// Escapes text for a quoted DOT string.
std::string quoted(const std::string& s) {
	std::string out = "\"";
	for (char c : s) {
		if (c == '"' || c == '\\') out += '\\';
		if (c == '\n') {
			out += "\\n";
			continue;
		}
		out += c;
	}
	return out + "\"";
}

// Escapes text for a DOT HTML-like label.
std::string html(const std::string& s) {
	std::string out;
	for (char c : s) {
		switch (c) {
			case '&': out += "&amp;"; break;
			case '<': out += "&lt;"; break;
			case '>': out += "&gt;"; break;
			case '"': out += "&quot;"; break;
			default: out += c;
		}
	}
	return out;
}

const char* fill_color(OpKind op) {
	switch (op) {
		case OpKind::Input: return "#dbeafe";
		case OpKind::Constant: return "#e5e7eb";
		case OpKind::MatMul: return "#fed7aa";
		case OpKind::FusedElementwise: return "#dcfce7";
		default: return "#fef3c7";
	}
}

std::string fused_label(const Graph& graph, const Node& n, bool show_body) {
	const FusedProgram& p = *n.fused;
	std::ostringstream os;
	os << "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"1\">"
	   << "<TR><TD><B>" << html(n.name) << "</B></TD></TR>"
	   << "<TR><TD>fused: " << p.num_ops() << " ops</TD></TR>"
	   << "<TR><TD>" << html(to_string(n.type)) << "</TD></TR>";
	if (show_body) {
		os << "<TR><TD ALIGN=\"LEFT\"><FONT FACE=\"Courier\" POINT-SIZE=\"9\">";
		for (std::size_t i=0; i<p.instrs.size(); ++i) {
			const FusedInstr& ins = p.instrs[i];
			std::ostringstream line;
			line << "r" << i << " = ";
			switch (ins.kind) {
				case FusedInstr::Kind::Load: line << "load " << graph.node(n.inputs[ins.a]).name; break;
				case FusedInstr::Kind::Immediate: line << ins.immediate; break;
				case FusedInstr::Kind::Unary: line << op_name(ins.op) << " r" << ins.a; break;
				case FusedInstr::Kind::Binary: line << op_name(ins.op) << " r" << ins.a << ", r" << ins.b; break;
			}
			os << html(line.str()) << "<BR ALIGN=\"LEFT\"/>";
		}
		os << "</FONT></TD></TR>";
	}
	os << "</TABLE>>";
	return os.str();
}

}

std::string to_dot(const Graph& graph, const DotOptions& options) {
	const GraphStats stats = compute_stats(graph);
	const std::string title = options.title.empty() ? graph.name() : options.title;

	std::ostringstream os;
	os << "digraph " << quoted(graph.name()) << " {\n"
	   << "  graph [rankdir=TB, fontname=\"Helvetica\", fontsize=13, labelloc=t, label="
	   << quoted(title + "\n" + std::to_string(stats.nodes) + " nodes, " + std::to_string(stats.compute_nodes) +
	             " compute (" + std::to_string(stats.elementwise_ops) + " elementwise ops, " +
	             std::to_string(stats.matmuls) + " matmul)")
	   << "];\n"
	   << "  node [fontname=\"Helvetica\", fontsize=10, shape=box, style=\"rounded,filled\"];\n"
	   << "  edge [color=\"#4b5563\", arrowsize=0.7];\n";

	for (std::size_t i=0; i<graph.num_nodes(); ++i) {
		const NodeId id = static_cast<NodeId>(i);
		const Node& n = graph.node(id);
		os << "  n" << id << " [";
		if (n.op == OpKind::FusedElementwise) {
			os << "label=" << fused_label(graph, n, options.show_fused_bodies) << ", color=\"#16a34a\"";
		} else {
			std::string label = n.name + "\n";
			if (n.op == OpKind::Constant) {
				label += "const " + describe_constant(n) + "\n";
			} else {
				label += std::string(op_name(n.op)) + "\n";
			}
			label += to_string(n.type);
			os << "label=" << quoted(label);
			if (n.op == OpKind::Input) os << ", shape=ellipse, style=filled";
			if (n.op == OpKind::Constant) os << ", style=filled";
		}
		os << ", fillcolor=\"" << fill_color(n.op) << "\"";
		if (graph.is_output(id)) os << ", peripheries=2, penwidth=1.5";
		os << "];\n";
	}

	for (std::size_t i=0; i<graph.num_nodes(); ++i) {
		const Node& n = graph.node(static_cast<NodeId>(i));
		NodeId previous = kNoNode;
		for (NodeId in : n.inputs) {
			if (in == previous) continue;  // x*x: draw one edge
			os << "  n" << in << " -> n" << i << ";\n";
			previous = in;
		}
	}
	os << "}\n";
	return os.str();
}

Status write_dot(const Graph& graph, const std::string& path, const DotOptions& options) {
	std::ofstream out(path, std::ios::binary);
	if (!out) return Error{"cannot write '" + path + "'"};
	out << to_dot(graph, options);
	if (!out) return Error{"failed writing '" + path + "'"};
	return Status();
}

}
