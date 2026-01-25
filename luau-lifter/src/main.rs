fn main() {
    let file_name = std::env::args().nth(1).expect("expected exactly one file");
    let key = std::env::args()
        .nth(2)
        .map(|s| {
            if s == "-e" {
                203
            } else if s == "-d" {
                // Dump mode - return early after dumping
                0
            } else {
                s.parse::<u8>().unwrap_or(1)
            }
        })
        .unwrap_or(1);

    let bytecode = std::fs::read(&file_name).expect("failed to read file");

    // Check for dump mode
    if std::env::args().nth(2).as_deref() == Some("-d") {
        let func_name = std::env::args().nth(3);
        luau_lifter::dump_bytecode(&bytecode, 1, func_name.as_deref());
        return;
    }

    println!("{}", luau_lifter::decompile_bytecode(&bytecode, key));
}
