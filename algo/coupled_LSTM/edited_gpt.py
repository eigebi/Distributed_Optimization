import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List

def _pack_vars(rho: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return np.concatenate([rho.flatten(), eta.flatten()])

def _unpack_slice_vec(x_slice: np.ndarray, B: int) -> Tuple[np.ndarray, np.ndarray]:
    rho_s = x_slice[:B].copy()
    eta_s = x_slice[B:].copy()
    return rho_s, eta_s

def _merge_slices_to_full(x_all: List[np.ndarray], S: int, B: int) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.zeros((S,B), dtype=np.float32)
    eta = np.zeros((S,B), dtype=np.float32)
    for s in range(S):
        rho_s, eta_s = _unpack_slice_vec(x_all[s], B)
        rho[s,:] = rho_s
        eta[s,:] = eta_s
    return rho, eta

def _split_full_to_slices(rho: np.ndarray, eta: np.ndarray) -> List[np.ndarray]:
    S, B = rho.shape
    out = []
    for s in range(S):
        out.append(np.concatenate([rho[s,:], eta[s,:]], axis=0).astype(np.float32))
    return out

def my_train_true_gradient(paras, prob, model, optimizer):
    (num_epoch, num_frame, num_iteration, num_var, S, len_lambda, num_problem, arg_nn) = paras
    assert S == 3, "This trainer expects S=3 slices (H, L, M)."

    x_models, lambda_models = model
    x_optimizers, lambda_optimizers = optimizer

    device = torch.device("cpu")
    for m in x_models + lambda_models:
        m.to(device)

    B = num_var // 2

    def obj_and_grads(rho, eta):
        f, grad_rho, grad_eta, extras = prob.objective_and_grad(rho, eta)
        return f, grad_rho, grad_eta, extras

    def cons_and_jacs(rho, eta):
        return prob.constraints_and_jacobians(rho, eta)

    def init_x_from_feasible_guess():
        a_sb = prob.d.a_sb
        rho0 = np.where(a_sb > 0, 1.0 / np.maximum(1, np.sum(a_sb, axis=0, keepdims=True)), 0.0).astype(np.float32)
        eta0 = rho0.copy()
        return _split_full_to_slices(rho0, eta0)

    a_sb = prob.d.a_sb.astype(np.float32)
    lb_rho = np.zeros((S,B), dtype=np.float32)
    ub_rho = a_sb.copy()
    lb_eta = np.zeros((S,B), dtype=np.float32)
    ub_eta = a_sb.copy()

    def project_box(rho, eta):
        rho = np.minimum(np.maximum(rho, lb_rho), ub_rho)
        eta = np.minimum(np.maximum(eta, lb_eta), ub_eta)
        return rho, eta

    step_x = 1.0
    step_l = 1.0

    L_train_result = []
    Loss_train_result = []
    L_truth_result = []
    obj_truth_result = []

    hx = [None, None, None]
    hl = [None, None, None]

    for epoch in range(num_epoch):
        x_all = init_x_from_feasible_guess()
        lam_H = np.abs(np.random.randn(1).astype(np.float32))
        lam_L = np.abs(np.random.randn(B).astype(np.float32))
        lam_R = np.abs(np.random.randn(2).astype(np.float32))

        running_loss = 0.0

        for frame in range(num_frame):
            for it in range(num_iteration):
                rho, eta = _merge_slices_to_full(x_all, S, B)
                rho, eta = project_box(rho, eta)

                f, grad_rho, grad_eta, extras = obj_and_grads(rho, eta)
                cons = cons_and_jacs(rho, eta)

                h_H = cons['h_H']; res_H = max(0.0, float(-h_H))
                h_L = cons['h_L']; res_L = np.maximum(0.0, (-h_L).astype(np.float32))
                g_rho = cons['g_rho']; g_eta = cons['g_eta']
                vio_rho = max(0.0, float(np.max(g_rho)))
                vio_eta = max(0.0, float(np.max(g_eta)))
                res_R = np.array([vio_rho, vio_eta], dtype=np.float32)

                base_grad_rho = -grad_rho
                base_grad_eta = -grad_eta

                base_grad_rho += lam_H[0] * (-cons['grad_hH_rho'])
                base_grad_eta += lam_H[0] * (-cons['grad_hH_eta'])

                for b in range(B):
                    base_grad_rho += lam_L[b] * ( - cons['grad_hL_rho'][b] )
                    base_grad_eta += lam_L[b] * ( - cons['grad_hL_eta'][b] )

                bstar_rho = int(np.argmax(g_rho))
                bstar_eta = int(np.argmax(g_eta))
                Jg_rho = cons['J_g_rho']
                Jg_eta = cons['J_g_eta']
                add_rho = np.zeros_like(base_grad_rho)
                add_eta = np.zeros_like(base_grad_eta)
                add_rho[:, bstar_rho] = lam_R[0] * Jg_rho[:, bstar_rho]
                add_eta[:, bstar_eta] = lam_R[1] * Jg_eta[:, bstar_eta]
                base_grad_rho += add_rho
                base_grad_eta += add_eta

                x_grads = []
                for s in range(S):
                    g_slice = np.concatenate([base_grad_rho[s,:], base_grad_eta[s,:]], axis=0).astype(np.float32)
                    x_grads.append(g_slice)

                new_x_all = []
                for s in range(S):
                    grad_in = torch.tensor(x_grads[s][None, None, :], dtype=torch.float32, device=device)
                    delta, hx_s = x_models[s](grad_in, hx[s])
                    hx[s] = (hx_s[0].detach(), hx_s[1].detach())
                    x_vec = torch.tensor(x_all[s][None, None, :], dtype=torch.float32, device=device) + delta[0]
                    new_x_all.append(x_vec.detach().cpu().numpy().reshape(-1))
                x_all = new_x_all

                lam_inputs = [
                    np.array([res_H], dtype=np.float32),
                    res_L.astype(np.float32),
                    res_R.astype(np.float32)
                ]
                lam_states = [lam_H, lam_L, lam_R]
                for i in range(3):
                    grad_lam = torch.tensor(lam_inputs[i][None, None, :], dtype=torch.float32, device=device)
                    delta_lam, hl_i = lambda_models[i](grad_lam, hl[i])
                    hl[i] = (hl_i[0].detach(), hl_i[1].detach())
                    lam_vec = torch.tensor(lam_states[i][None, :], dtype=torch.float32, device=device) + delta_lam[0]
                    lam_states[i] = torch.relu(lam_vec).detach().cpu().numpy().reshape(-1)
                lam_H, lam_L, lam_R = lam_states

                J_val = (-f + lam_H[0]*res_H + float(np.dot(lam_L, res_L)) + lam_R[0]*vio_rho + lam_R[1]*vio_eta)
                running_loss += J_val

                for opt in x_optimizers: opt.zero_grad()
                for opt in lambda_optimizers: opt.zero_grad()
                loss_t = torch.tensor(J_val, dtype=torch.float32, device=device)
                loss_t.backward()
                for opt in x_optimizers: opt.step()
                for opt in lambda_optimizers: opt.step()

        L_train_result.append([np.linalg.norm(lam_H), np.linalg.norm(lam_L), np.linalg.norm(lam_R)])
        Loss_train_result.append(running_loss/(num_frame*num_iteration))

        rho_chk, eta_chk = _merge_slices_to_full(x_all, S, B)
        rho_chk, eta_chk = project_box(rho_chk, eta_chk)
        f_chk, _, _, _ = obj_and_grads(rho_chk, eta_chk)
        cons_chk = cons_and_jacs(rho_chk, eta_chk)
        obj_truth_result.append(f_chk)
        L_truth_result.append([float(np.max(cons_chk['g_rho'])),
                               float(np.max(cons_chk['g_eta'])),
                               float(-cons_chk['h_H']),
                               float(np.max(-cons_chk['h_L']))])

        print(f"[Epoch {epoch+1}/{num_epoch}] J={running_loss/(num_frame*num_iteration):.4f}  "
              f"f={f_chk:.4f}  max(g_rho)={L_truth_result[-1][0]:.3e}  "
              f"max(g_eta)={L_truth_result[-1][1]:.3e}  -h_H={L_truth_result[-1][2]:.3e}  "
              f"max(-h_L)={L_truth_result[-1][3]:.3e}")

    return L_train_result, Loss_train_result, L_truth_result, obj_truth_result
