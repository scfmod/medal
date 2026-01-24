fn main() {
    let file_name = std::env::args().nth(1).expect("expected exactly one file");
    let key = std::env::args()
        .nth(2)
        .map(|s| {
            if s == "-e" {
                203
            } else {
                s.parse::<u8>().unwrap_or(1)
            }
        })
        .unwrap_or(1);
    let bytecode = std::fs::read(file_name).expect("failed to read file");
    println!("{}", luau_lifter::decompile_bytecode(&bytecode, key));
}
