#pragma once

#include "llama.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);
void                    common_sampler_copy  (const struct common_sampler * src, struct common_sampler * dst);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

// [CGC §8.108] L4-pool variant: accept externally-cached logits instead of reading from ctx.
// Used when L4 metal pool caps n_batch=1 and the main + draft tokens span multiple sub-batches,
// so the logits cannot be read from ctx in a single call. The caller must cache the logits of
// each spec_i_batch token across sub-batches and pass them here in the same order as idxs.
// external_logits[i] must point to a buffer of n_vocab floats (n_vocab taken from ctx's model).
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<const float *> & external_logits, bool grammar_first = false);

// [CGC M4 rejection sampling 2026-09-17] The draft's own distribution at one drafted position.
//
// WHY THIS EXISTS: the accept rule used to be `draft[i] == id` -- exact token equality. Under that
// rule P(accept) = sum_e p_t(e)*p_d(e), which is MAXIMISED when p_d is one-hot (the argmax). So no
// amount of draft-sampler tuning could raise accept, and the 2026-09-13 sampler-parity work (which
// made the draft actually DRAW from its own distribution instead of forcing the argmax) correctly
// LOWERED it -- the comment at that site already said parity is "a prerequisite for rejection
// sampling, not a speedup on its own". This is the other half: accept x with probability
// min(1, p_t(x)/q(x)); on rejection, emit a draw from the residual max(0, p_t - q).
//
// The MTP draft's sampler chain is mirrored from the target's, so p_t and q are restricted the same
// way (both are post-chain candidate arrays, not full model softmaxes). The ratio below is therefore
// a ratio of two comparable quantities -- it is NOT a ratio of full distributions, and this note
// says so rather than letting a later reader assume it.
struct common_draft_dist {
    std::vector<llama_token> ids;
    std::vector<float>       probs;   // same length as ids; need not be normalised
};

// True when CGC_MTP_REJECTION is set. Exposed so a caller can decline to populate the draft
// distributions at all when the rule is off: the copy is small, but a small NONZERO cost in the
// default path is precisely how a "no-op" change stops being a no-op.
bool common_sampler_mtp_rejection_on();

// Rejection-sampling variant. `draft_dist` may be nullptr or partial: any position without a usable
// draft distribution falls back to the exact-match rule, so a caller that carries distributions for
// only some positions still gets a well-defined answer. Everything is additionally gated on
// CGC_MTP_REJECTION so a default run cannot reach any of it.
//
// Both the ctx-logits path and the external-logits path have one of these, so the rule is the same
// wherever a draft is verified -- an asymmetry between the two would make the two instruments
// incomparable for exactly the reason M4 is trying to fix.
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<common_draft_dist> * draft_dist, bool grammar_first = false);

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<const float *> & external_logits, const std::vector<common_draft_dist> * draft_dist, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
