use cfg::dot;
use cfg::function::Function;
use fxhash::FxHashMap;
use graph::{algorithms::*, Edge, Graph, NodeId, Directed};

mod compound;
mod conditional;
mod jump;
mod r#loop;

struct GraphStructurer {
    function: Function,
    root: NodeId,
    back_edges: Vec<Edge>,
}

impl GraphStructurer {
    fn new(
        function: Function,
        graph: Graph<Directed>,
        blocks: FxHashMap<NodeId, ast::Block>,
        root: NodeId,
    ) -> Self {
        let back_edges = back_edges(&graph, root).unwrap();
        let root = function.entry().unwrap();
        Self {
            function,
            root,
            back_edges,
        }
    }

    fn block_is_no_op(block: &ast::Block) -> bool {
        block
            .iter()
            .filter(|stmt| stmt.as_comment().is_some())
            .count()
            == block.len()
    }

    fn try_match_pattern(&mut self, node: NodeId) -> bool {
        let successors = self.function.graph().successors(node);

        if self.try_collapse_loop(node) {
            return true;
        }

        let changed = match successors.len() {
            0 => false,
            1 => {
                // remove unnecessary jumps to allow pattern matching
                self.match_jump(node, successors[0])
            }
            2 => {
                let (then_edge, else_edge) = self
                    .function
                    .block(node)
                    .unwrap()
                    .terminator
                    .as_ref()
                    .unwrap()
                    .as_conditional()
                    .unwrap();
                let (then_node, else_node) = (then_edge.node, else_edge.node);
                //self.match_conditional(node, then_node, else_node);
                self.match_compound_conditional(node, then_node, else_node)
            }

            _ => unreachable!(),
        };

        //dot::render_to(&self.function, &mut std::io::stdout());

        changed
    }

    fn match_blocks(&mut self) -> bool {
        let dfs = dfs_tree(self.function.graph(), self.root);
        for node in self
            .function
            .graph()
            .nodes()
            .iter()
            .filter(|&&node| !dfs.has_node(node))
            .cloned()
            .collect::<Vec<_>>()
        {
            self.function.remove_block(node);
        }

        let mut changed = false;
        for node in dfs.post_order(self.root) {
            println!("matching {}", node);
            changed |= self.try_match_pattern(node);
        }

        cfg::dot::render_to(&self.function, &mut std::io::stdout()).unwrap();

        changed
    }

    fn collapse(&mut self) {
        while self.match_blocks() {}

        let nodes = self.function.graph().nodes().len();
        if self.function.graph().nodes().len() != 1 {
            println!("failed to collapse! total nodes: {}", nodes);
        }
    }

    fn structure(mut self) -> ast::Block {
        self.collapse();
        self.function.remove_block(self.root).unwrap().ast
    }
}

pub fn lift(function: cfg::function::Function) {
    let graph = function.graph().clone();
    let root = function.entry().unwrap();

    //dot::render_to(&graph, &mut std::io::stdout());

    let blocks = function
        .blocks()
        .iter()
        .map(|(&node, block)| (node, block.ast.clone()))
        .collect();

    let structurer = GraphStructurer::new(function, graph, blocks, root);
    let block = structurer.structure();
    println!("{}", block);
}
