#include "jsmn/jsmn.h"

///**
// * Allocates a fresh unused token from the token pull.
// */
//static jsmntok_t *jsmn_alloc_token(jsmn_parser *parser,
//		jsmntok_t *tokens, size_t num_tokens) {
//	jsmntok_t *tok;
//	if (parser->toknext >= num_tokens) {
//		/* */
//#ifndef JSON_TPCH_WIDE
//		return NULL;
//#endif
//#ifdef JSON_TPCH_WIDE
//		return NULL;
//#endif
//	}
//	tok = &tokens[parser->toknext++];
//	tok->start = tok->end = -1;
//	tok->size = 0;
//#ifdef JSMN_PARENT_LINKS
//	tok->parent = -1;
//#endif
//	return tok;
//}

/**
 * Allocates a fresh unused token from the token pool.
 */
static jsmntok_t *jsmn_alloc_token(jsmn_parser *parser,
		jsmntok_t **tokens, size_t *num_tokens) {
	jsmntok_t *tok;
	if (parser->toknext >= *num_tokens) {

		/* REALLOC */
#if defined(JSON_TPCH_WIDE) || defined(JSON_SYMANTEC_WIDE)
		//printf("Increasing size for large json object\n");
		void *oldRegion = (void*) *tokens;
		size_t oldTokenNo = *num_tokens;
		size_t newTokenNo = oldTokenNo << 1;

		//printf("check: %d\n",tokens[0][0].end);
		size_t oldSize = oldTokenNo * sizeof(jsmntok_t);
#ifdef DEBUGJSMN
		size_t newSize = newTokenNo * sizeof(jsmntok_t);
		{
			printf("New size: %ld %ld\n", newTokenNo, newSize);
		}
#endif

		//void* newRegion = realloc(oldRegion, newSize);
		void *newRegion = calloc(newTokenNo, sizeof(jsmntok_t));
		if (newRegion != NULL ) {
#ifdef DEBUGJSMN
			{
				printf("Calloc gave me %ld %ld - before it was %ld\n", tokens,
						newRegion, oldRegion);
			}
#endif
			memcpy(newRegion, oldRegion, oldSize);
			*tokens = (jsmntok_t*) newRegion;
			*num_tokens = newTokenNo;
			//free(oldRegion);
		} else {
			return NULL ;
		}
#else
		return NULL;
#endif

	}
	unsigned int nextTokenNo = parser->toknext++;
	tok = &((*tokens)[nextTokenNo]);
	tok->start = tok->end = -1;
	tok->size = 0;
#ifdef DEBUGJSMN
	{
		printf("Providing token %u\n", nextTokenNo);
		printf("Token info: %ld %d %d %d\n", tok, tok->start, tok->end,
				tok->size);
	}
#endif
	return tok;
}

/**
 * Fills token type and boundaries.
 */
static void jsmn_fill_token(jsmntok_t *token, jsmntype_t type,
                            int start, int end) {
	token->type = type;
	token->start = start;
	token->end = end;
	token->size = 0;
#ifdef DEBUGJSMN
	{
		printf("[jsmn_fill_token: ] Token filled\n");
	}
#endif
}

/**
 * Fills next available token with JSON primitive.
 */
static jsmnerr_t jsmn_parse_primitive(jsmn_parser *parser, const char *js,
		size_t len, jsmntok_t **tokens, size_t *num_tokens) {
	jsmntok_t *token;
	int start;

	start = parser->pos;

	for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
		switch (js[parser->pos]) {
#ifndef JSMN_STRICT
			/* In strict mode primitive must be followed by "," or "}" or "]" */
			case ':':
#endif
			case '\t' : case '\r' : case '\n' : case ' ' :
			case ','  : case ']'  : case '}' :
				goto found;
		}
		if (js[parser->pos] < 32 || js[parser->pos] >= 127) {
			parser->pos = start;
#ifdef DEBUGJSMN
			{
				printf("Invalid character at position %u when parsing primitive\n", parser->pos);
			}
#endif
			return JSMN_ERROR_INVAL;
		}
	}
#ifdef JSMN_STRICT
	/* In strict mode primitive must be followed by a comma/object/array */
	parser->pos = start;
	return JSMN_ERROR_PART;
#endif

found:
	if (*tokens == NULL) {
		parser->pos--;
		return 0;
	}
	token = jsmn_alloc_token(parser, tokens, num_tokens);
	if (token == NULL) {
		parser->pos = start;
		return JSMN_ERROR_NOMEM;
	}
#ifdef DEBUGJSMN
	{
		printf("1. (Token to fill info): %ld %d %d %d\n", token, token->start,
				token->end, token->size);
	}
#endif
	jsmn_fill_token(token, JSMN_PRIMITIVE, start, parser->pos);
#ifdef DEBUGJSMN
	{
		printf("1. Token filled\n");
	}
#endif

#ifdef JSMN_PARENT_LINKS
	token->parent = parser->toksuper;
#endif
	parser->pos--;
	return 0;
}

/**
 * Fills next token with JSON string.
 */
static jsmnerr_t jsmn_parse_string(jsmn_parser *parser, const char *js,
		size_t len, jsmntok_t **tokens, size_t *num_tokens) {
	jsmntok_t *token;

	int start = parser->pos;

	parser->pos++;

	/* Skip starting quote */
	for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
		char c = js[parser->pos];

		/* Quote: end of string */
		if (c == '\"') {
			if (*tokens == NULL) {
				return 0;
			}
			token = jsmn_alloc_token(parser, tokens, num_tokens);
			if (token == NULL) {
				parser->pos = start;
				return JSMN_ERROR_NOMEM;
			}
#ifdef DEBUGJSMN
	{
		printf("2. (Token to fill info): %ld %d %d %d\n", token, token->start,
				token->end, token->size);
	}
#endif
			jsmn_fill_token(token, JSMN_STRING, start+1, parser->pos);
#ifdef DEBUGJSMN
	{
		printf("2. Token filled\n");
	}
#endif
#ifdef JSMN_PARENT_LINKS
			token->parent = parser->toksuper;
#endif
			return 0;
		}

		/* Backslash: Quoted symbol expected */
		if (c == '\\') {
			parser->pos++;
			switch (js[parser->pos]) {
				/* Allowed escaped symbols */
				case '\"': case '/' : case '\\' : case 'b' :
				case 'f' : case 'r' : case 'n'  : case 't' :
					break;
				/* Allows escaped symbol \uXXXX */
				case 'u':
					parser->pos++;
					int i = 0;
					for(; i < 4 && js[parser->pos] != '\0'; i++) {
						/* If it isn't a hex character we have an error */
						if(!((js[parser->pos] >= 48 && js[parser->pos] <= 57) || /* 0-9 */
									(js[parser->pos] >= 65 && js[parser->pos] <= 70) || /* A-F */
									(js[parser->pos] >= 97 && js[parser->pos] <= 102))) { /* a-f */
							parser->pos = start;
#ifdef DEBUGJSMN
			{
				printf("Invalid character at position %u - char not hex\n", parser->pos);
			}
#endif
							return JSMN_ERROR_INVAL;
						}
						parser->pos++;
					}
					parser->pos--;
					break;
				/* Unexpected symbol */
				default:
					parser->pos = start;
#ifdef DEBUGJSMN
			{
				printf("Invalid character at position %u - unexpected symbol\n", parser->pos);
			}
#endif
					return JSMN_ERROR_INVAL;
			}
		}
	}
	parser->pos = start;
	return JSMN_ERROR_PART;
}

/**
 * Parse JSON string and fill tokens.
 */
/**
 * Parse JSON string and fill tokens.
 */
jsmnerr_t jsmn_parse(jsmn_parser *parser, const char *js, size_t len,
		jsmntok_t **tokensPtr, size_t num_tokens) {
	jsmnerr_t r;
	int i;
	jsmntok_t *token;
	int count = 0;

	for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
		char c;
		jsmntype_t type;

		c = js[parser->pos];
		switch (c) {
		case '{':
		case '[':
			count++;
			if (*tokensPtr == NULL ) {
				break;
			}
			token = jsmn_alloc_token(parser, tokensPtr, &num_tokens);
#ifdef DEBUGJSMN
			{
				printf("3. (Token to fill info): %ld %d %d %d\n", token,
						token->start, token->end, token->size);
			}
#endif
			if (token == NULL )
				return JSMN_ERROR_NOMEM;
			if (parser->toksuper != -1) {
				(*tokensPtr)[parser->toksuper].size++;
#ifdef JSMN_PARENT_LINKS
				token->parent = parser->toksuper;
#endif
			}
			token->type = (c == '{' ? JSMN_OBJECT : JSMN_ARRAY);
			token->start = parser->pos;
			parser->toksuper = parser->toknext - 1;
			break;
		case '}':
		case ']':
			if ((*tokensPtr) == NULL )
				break;
			type = (c == '}' ? JSMN_OBJECT : JSMN_ARRAY);
#ifdef JSMN_PARENT_LINKS
			if (parser->toknext < 1) {
				return JSMN_ERROR_INVAL;
			}
			token = &tokens[parser->toknext - 1];
			for (;;) {
				if (token->start != -1 && token->end == -1) {
					if (token->type != type) {
						return JSMN_ERROR_INVAL;
					}
					token->end = parser->pos + 1;
					parser->toksuper = token->parent;
					break;
				}
				if (token->parent == -1) {
					break;
				}
				token = &tokens[token->parent];
			}
#else
			for (i = parser->toknext - 1; i >= 0; i--) {
				token = &(*tokensPtr)[i];
				if (token->start != -1 && token->end == -1) {
					if (token->type != type) {
#ifdef DEBUGJSMN
						{
							printf(
									"Invalid character at position %u - wrong type\n",
									parser->pos);
						}
#endif
						return JSMN_ERROR_INVAL;
					}
					parser->toksuper = -1;
					token->end = parser->pos + 1;
					break;
				}
			}
			/* Error if unmatched closing bracket */
			if (i == -1)
			{
#ifdef DEBUGJSMN
				{
					printf(
							"Invalid character at position %u - unmatched bracket\n",
							parser->pos);
				}
#endif
				return JSMN_ERROR_INVAL;
			}
			for (; i >= 0; i--) {
				token = &(*tokensPtr)[i];
				if (token->start != -1 && token->end == -1) {
					parser->toksuper = i;
					break;
				}
			}
#endif
			break;
		case '\"':
			r = jsmn_parse_string(parser, js, len, &(*tokensPtr), &num_tokens);
			if (r < 0)
				return r;
			count++;
			if (parser->toksuper != -1 && (*tokensPtr) != NULL )
				(*tokensPtr)[parser->toksuper].size++;
			break;
		case '\t':
		case '\r':
		case '\n':
		case ':':
		case ',':
		case ' ':
			break;
#ifdef JSMN_STRICT
			/* In strict mode primitives are: numbers and booleans */
			case '-': case '0': case '1' : case '2': case '3' : case '4':
			case '5': case '6': case '7' : case '8': case '9':
			case 't': case 'f': case 'n' :
#else
			/* In non-strict mode every unquoted value is a primitive */
		default:
#endif
			r = jsmn_parse_primitive(parser, js, len, tokensPtr, &num_tokens);
			if (r < 0)
				return r;
			count++;
			if (parser->toksuper != -1 && (*tokensPtr) != NULL )
			{
//				printf("1. %d\n",parser->toksuper);
//				printf("2. %d\n",(*tokensPtr)[parser->toksuper].size);
				(*tokensPtr)[parser->toksuper].size++;
			}
			break;

#ifdef JSMN_STRICT
			/* Unexpected char in strict mode */
			default:
			return JSMN_ERROR_INVAL;
#endif
		}
	}

	for (i = parser->toknext - 1; i >= 0; i--) {
		/* Unmatched opened object or array */
		if ((*tokensPtr)[i].start != -1 && (*tokensPtr)[i].end == -1) {
			return JSMN_ERROR_PART;
		}
	}
	//printf("[jsmn: ] %d to %d with size %d\n",(*tokensPtr)[0].start,(*tokensPtr)[0].end, (*tokensPtr)[0].size);

	/* Useful to find number of tokens per line */
	/* printf("TokenNo: %u\n", parser->toknext); */
	return count;
}

/**
 * Creates a new parser based over a given  buffer with an array of tokens
 * available.
 */
void jsmn_init(jsmn_parser *parser) {
	parser->pos = 0;
	parser->toknext = 0;
	parser->toksuper = -1;
}
