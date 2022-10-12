use std::{borrow::Cow, io::Write};

use dot::{GraphWalk, LabelText, Labeller};

use petgraph::stable_graph::{EdgeIndex, NodeIndex};

use crate::function::Function;

fn arguments(args: &Vec<(ast::RcLocal, ast::RcLocal)>) -> String {
    let mut s = String::new();
    for (i, (local, new_local)) in args.iter().enumerate() {
        use std::fmt::Write;
        write!(s, "{} -> {}", local, new_local).unwrap();
        if i + 1 != args.len() {
            s.push('\n');
        }
    }
    s
}

impl<'a> Labeller<'a, NodeIndex, EdgeIndex> for Function {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("cfg").unwrap()
    }

    fn node_label<'b>(&'b self, n: &NodeIndex) -> dot::LabelText<'b> {
        let block = self.block(*n).unwrap();
        dot::LabelText::LabelStr(block.to_string().into())
            .prefix_line(dot::LabelText::LabelStr(n.index().to_string().into()))
    }

    fn edge_label<'b>(&'b self, e: &EdgeIndex) -> dot::LabelText<'b> {
        let edge = self.graph().edge_weight(*e).unwrap();
        match edge.branch_type {
            crate::block::BranchType::Unconditional => {
                dot::LabelText::LabelStr(arguments(&edge.arguments).into())
            }
            crate::block::BranchType::Then => {
                let arguments = arguments(&edge.arguments);
                if !arguments.is_empty() {
                    dot::LabelText::LabelStr(format!("t\n{}", arguments).into())
                } else {
                    dot::LabelText::LabelStr("t".into())
                }
            }
            crate::block::BranchType::Else => {
                let arguments = arguments(&edge.arguments);
                if !arguments.is_empty() {
                    dot::LabelText::LabelStr(format!("e\n{}", arguments).into())
                } else {
                    dot::LabelText::LabelStr("e".into())
                }
            }
        }
    }

    fn node_id(&'a self, n: &NodeIndex) -> dot::Id<'a> {
        dot::Id::new(format!("N{}", n.index())).unwrap()
    }

    fn node_shape(&'a self, _n: &NodeIndex) -> Option<LabelText<'a>> {
        Some(LabelText::LabelStr("rect".into()))
    }
}

impl<'a> GraphWalk<'a, NodeIndex, EdgeIndex> for Function {
    fn nodes(&'a self) -> dot::Nodes<'a, NodeIndex> {
        Cow::Owned(self.graph().node_indices().collect())
    }

    fn edges(&'a self) -> dot::Edges<'a, EdgeIndex> {
        Cow::Owned(self.graph().edge_indices().collect())
    }

    fn source(&self, e: &EdgeIndex) -> NodeIndex {
        self.graph().edge_endpoints(*e).unwrap().0
    }

    fn target(&self, e: &EdgeIndex) -> NodeIndex {
        self.graph().edge_endpoints(*e).unwrap().1
    }
}

pub fn render_to<W: Write>(cfg: &Function, output: &mut W) -> std::io::Result<()> {
    dot::render(cfg, output)
}
