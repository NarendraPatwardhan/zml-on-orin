use clap::{Parser, Subcommand};

mod engine;

#[derive(Parser)]
#[command(name = "zml_on_arm64", about = "ZML on a Jetson Orin Nano")]
struct Cli {
    /// Print ZML and PJRT info logs.
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Multiply two small matrices on the CUDA platform and check the result.
    Matmul,
}

fn main() {
    let cli = Cli::parse();
    engine::set_verbose(cli.verbose);

    let result = match cli.command {
        Commands::Matmul => prove_matmul(),
    };

    if let Err(err) = result {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn prove_matmul() -> Result<(), String> {
    // [[1, 2, 3], [4, 5, 6]] × [[7, 8], [9, 10], [11, 12]]
    // = [[58, 64], [139, 154]]
    const M: u32 = 2;
    const K: u32 = 3;
    const N: u32 = 2;
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = [58.0_f32, 64.0, 139.0, 154.0];

    let engine = engine::Engine::new()?;
    let platform = engine.platform_name();
    println!("platform: {platform}");
    if platform != "cuda" {
        return Err(format!("expected the cuda platform, got {platform}"));
    }

    let got = engine.matmul(M, K, N, &a, &b)?;
    println!("result: {got:?}");
    if got.len() != expected.len() || got.iter().zip(expected).any(|(g, e)| *g != e) {
        return Err(format!("expected {expected:?}, got {got:?}"));
    }
    println!("matmul ok");
    Ok(())
}
