use crate::board::attacks;
use crate::board::bitboard::Bitboard;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::Board;
use crate::search::parameters::{
    see_value_bishop, see_value_knight, see_value_pawn, see_value_queen, see_value_rook,
};

pub fn value(pc: Piece) -> i32 {
    match pc {
        Piece::Pawn => see_value_pawn(),
        Piece::Knight => see_value_knight(),
        Piece::Bishop => see_value_bishop(),
        Piece::Rook => see_value_rook(),
        Piece::Queen => see_value_queen(),
        Piece::King => 0,
    }
}

pub fn see(board: &Board, mv: &Move, threshold: i32) -> bool {
    let from = mv.from();
    let to = mv.to();

    let next_victim = mv
        .promo_piece()
        .map_or_else(|| board.piece_at(from).unwrap(), |promo| promo);

    let mut balance = move_value(board, mv) - threshold;

    if balance < 0 {
        return false;
    }

    balance -= value(next_victim);

    if balance >= 0 {
        return true;
    }

    let mut occ = board.occ() ^ Bitboard::of_sq(from) ^ Bitboard::of_sq(to);

    if let Some(ep_sq) = board.ep_sq {
        occ ^= Bitboard::of_sq(ep_sq);
    }

    let mut attackers = attackers_to(board, to, occ) & occ;
    let diagonal = board.pcs(Piece::Bishop) | board.pcs(Piece::Queen);
    let orthogonal = board.pcs(Piece::Rook) | board.pcs(Piece::Queen);

    let mut stm = !board.stm;

    loop {
        let our_attackers = attackers & board.side(stm);
        if our_attackers.is_empty() {
            break;
        }

        let attacker = least_valuable_attacker(board, our_attackers);

        if attacker == Piece::King && !(attackers & board.side(!stm)).is_empty() {
            break;
        }

        // Make the capture
        let pcs = board.pcs(attacker) & our_attackers;
        let sq = (our_attackers & pcs).lsb();
        occ = occ.pop_bit(sq);
        stm = !stm;

        balance = -balance - 1 - value(attacker);
        if balance >= 0 {
            break;
        }

        // Capturing may reveal a new slider
        if [Piece::Pawn, Piece::Bishop, Piece::Queen].contains(&attacker) {
            attackers |= attacks::bishop(to, occ) & diagonal;
        }
        if [Piece::Rook, Piece::Queen].contains(&attacker) {
            attackers |= attacks::rook(to, occ) & orthogonal;
        }
        attackers &= occ;
    }

    stm != board.stm
}

#[allow(clippy::redundant_closure)]
fn move_value(board: &Board, mv: &Move) -> i32 {
    let mut see_value = board
        .piece_at(mv.to())
        .map_or(0, |captured| value(captured));

    if let Some(promo) = mv.promo_piece() {
        see_value += value(promo);
    } else if mv.is_ep() {
        see_value = value(Piece::Pawn);
    }
    see_value
}

// get a mask of the nonzero u64s in v
#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f")]
pub fn to_bitmasknz_stable(v: [u64; 8]) -> u8 {
    use std::arch::x86_64::*;
    unsafe {
        let x = _mm512_loadu_si512(v.as_ptr() as *const _);
        let k = _mm512_test_epi64_mask(x, x);
        k as u8
    }
}

// get a mask of the nonzero u64s in v
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
#[target_feature(enable = "avx2")]
pub fn to_bitmasknz_stable(v: [u64; 8]) -> u8 {
    use std::arch::x86_64::*;
    unsafe {
        let a = _mm256_loadu_si256(v.as_ptr() as *const _);
        let b = _mm256_loadu_si256(v.as_ptr().add(4) as *const _);
        let zero = _mm256_setzero_si256();

        let eq0_a = _mm256_cmpeq_epi64(a, zero);
        let eq0_b = _mm256_cmpeq_epi64(b, zero);

        let packed = _mm256_packs_epi32(eq0_a, eq0_b);
        let perm = _mm256_permute4x64_epi64::<0b11011000>(packed);
        let mask = _mm256_movemask_ps(std::mem::transmute(perm));
        !mask as u8
    }
}

fn least_valuable_attacker(board: &Board, our_attackers: Bitboard) -> Piece {
    // SAFETY: theyre wrapped u64s
    let mut bbs: [u64; 8] = unsafe { std::mem::transmute(board.bb) };
    bbs[6] = 0;
    bbs[7] = 0;

    let mut nz = [0; 8];
    for i in 0..8 {
        nz[i] = bbs[i] & our_attackers.0;
    }

    // SAFETY: we only support avx2 and avx512 targets so the function always exists
    return match unsafe { to_bitmasknz_stable(nz) }.trailing_zeros() {
        0 => Piece::Pawn,
        1 => Piece::Knight,
        2 => Piece::Bishop,
        3 => Piece::Rook,
        4 => Piece::Queen,
        5 => Piece::King,
        _ => unreachable!("invalid index for Piece"),
    };
}

fn attackers_to(board: &Board, square: Square, occupancies: Bitboard) -> Bitboard {
    let diagonals = board.pcs(Piece::Bishop) | board.pcs(Piece::Queen);
    let orthogonals = board.pcs(Piece::Rook) | board.pcs(Piece::Queen);
    let white_pawn_attacks = attacks::pawn(square, Side::Black) & board.pawns(Side::White);
    let black_pawn_attacks = attacks::pawn(square, Side::White) & board.pawns(Side::Black);
    let knight_attacks = attacks::knight(square) & board.pcs(Piece::Knight);
    let diagonal_attacks = attacks::bishop(square, occupancies) & diagonals;
    let orthogonal_attacks = attacks::rook(square, occupancies) & orthogonals;
    let king_attacks = attacks::king(square) & board.pcs(Piece::King);
    white_pawn_attacks
        | black_pawn_attacks
        | knight_attacks
        | diagonal_attacks
        | orthogonal_attacks
        | king_attacks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::movegen::MoveFilter;
    use crate::board::Board;
    use std::fs;

    // #[test]
    fn test_see_suite() {
        let see_suite = fs::read_to_string("../../resources/see.epd").unwrap();

        let mut tried = 0;
        let mut passed = 0;

        for see_test in see_suite.lines() {
            let parts: Vec<&str> = see_test.split("|").collect();
            let fen = parts[0].trim();
            let mv_uci = parts[1].trim();
            let threshold_str = parts[2].trim();
            let threshold: i32 = threshold_str.parse().unwrap();

            let board = Board::from_fen(fen).unwrap();
            let mut moves = board.gen_moves(MoveFilter::All);
            let mv = moves
                .iter()
                .map(|entry| entry.mv)
                .find(|m| m.to_uci() == mv_uci)
                .expect("Move not found in generated moves");

            tried += 1;
            if see(&board, &mv, threshold) {
                passed += 1;
            } else {
                println!("Failed SEE test for FEN: {} and move: {}", fen, mv_uci);
            }
        }

        assert_eq!(
            passed, tried,
            "Passed {} out of {} SEE tests",
            passed, tried
        );
    }
}
